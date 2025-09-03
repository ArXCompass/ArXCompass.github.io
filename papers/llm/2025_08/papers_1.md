# llm - 2025_08

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08325v2">CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17182v2">LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 This preprint is under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17117v4">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05702v2">Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06133v2">LLM Serving Optimization with Variable Prefill and Decode Lengths</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      We study the problem of serving LLM (Large Language Model) requests where each request has heterogeneous prefill and decode lengths. In LLM serving, the prefill length corresponds to the input prompt length, which determines the initial memory usage in the KV cache. The decode length refers to the number of output tokens generated sequentially, with each additional token increasing the KV cache memory usage by one unit. Given a set of n requests, our goal is to schedule and process them to minimize the total completion time. We show that this problem is NP-hard due to the interplay of batching, placement constraints, precedence relationships, and linearly increasing memory usage. We then analyze commonly used scheduling strategies in practice, such as First-Come-First-Serve (FCFS) and Shortest-First (SF), and prove that their competitive ratios scale up sublinearly with the memory limit-a significant drawback in real-world settings where memory demand is large. To address this, we propose a novel algorithm based on a new selection metric that efficiently forms batches over time. We prove that this algorithm achieves a constant competitive ratio. Finally, we develop and evaluate a few algorithm variants inspired by this approach, including dynamic programming variants, local search methods, and an LP-based scheduler, demonstrating through comprehensive simulations that they outperform standard baselines while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04078v3">LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01085v4">Explaining Length Bias in LLM-Based Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as judges, particularly in preference comparisons, has become widespread, but this reveals a notable bias towards longer responses, undermining the reliability of such evaluations. To better understand such bias, we propose to decompose the preference evaluation metric, specifically the win rate, into two key components: desirability and information mass, where the former is length-independent and related to trustworthiness such as correctness, toxicity, and consistency, and the latter is length-dependent and represents the amount of information in the response. We empirically demonstrated the decomposition through controlled experiments and found that response length impacts evaluations by influencing information mass. To derive a reliable evaluation metric that assesses content quality without being confounded by response length, we propose AdapAlpaca, a simple yet effective adjustment to win rate measurement. Specifically, AdapAlpaca ensures a fair comparison of response quality by aligning the lengths of reference and test model responses under equivalent length intervals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v4">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18798v2">Distill Visual Chart Reasoning Ability from LLMs to MLLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 Accepted to EMNLP 2025 Findings. The code and dataset are publicly available at https://github.com/hewei2001/ReachQA
    </div>
    <details class="paper-abstract">
      Solving complex chart Q&A tasks requires advanced visual reasoning abilities in multimodal large language models (MLLMs), including recognizing key information from visual inputs and conducting reasoning over it. While fine-tuning MLLMs for reasoning is critical, collecting and annotating charts and questions is expensive, hard to scale, and often results in low-quality annotations. To address this, we propose Code-as-Intermediary Translation (CIT), a cost-effective, efficient and scalable data synthesis method for distilling visual reasoning abilities from LLMs to MLLMs. The code serves as an intermediary that translates visual chart representations into textual representations, enabling language models to understand cross-modal information and generate reasoning chains accordingly. In this way, we can employ text-based synthesizing techniques to expand chart-plotting code and generate high-quality Q&A pairs for training models. This produces ReachQA, a dataset containing 3k reasoning-intensive charts and 20k Q&A pairs to enhance both recognition and reasoning abilities of MLLMs. Experiments show that models fine-tuned with ReachQA not only perform well on chart-related tasks but also show performance gains on general reasoning benchmarks. The code and dataset are publicly available at https://github.com/hewei2001/ReachQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12597v2">UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicle (UAV) individual (ID) identification is a critical security surveillance strategy in low-altitude integrated sensing and communication (ISAC) networks. In this paper, we propose a novel dynamic knowledge distillation (KD)-enabled wireless radio frequency fingerprint large language model (RFF-LLM) framework for UAV ID identification. First, we propose an RFF-LLM framework based on the modified GPT-2 model to improve the identification accuracy in complex outdoor environments. Then, considering the parameter overhead of the RFF-LLM, we design a dynamic KD strategy to compress the model. Specifically, the proximal policy optimization (PPO) algorithm is employed to dynamically adjust the distillation temperature, overcoming the local optimum dilemma inherent in static KD. As a next step, the knowledge of the RFF-LLM is adequately transferred to the lightweight Lite-HRNet model. Finally, our experiments are conducted based on the self-built drone RFF dataset of Release one, namely DRFF-R1, by collecting the I/Q signals of 20 commercial UAVs in channel 149. The experiment results show that the proposed framework achieves 98.38% ID identification accuracy with merely 0.15 million parameters and 2.74 ms response time, which outperforms the benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19030v3">RECAST: Strengthening LLMs' Complex Instruction Following with Constraint-Verifiable Data</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions. To address this challenge, we propose RECAST, a novel framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 15 constraint types. Experimental results demonstrate that models fine-tuned on RECAST-30K show substantial improvements in following complex instructions. Moreover, the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21471v2">LUMIR: an LLM-Driven Unified Agent Framework for Multi-task Infrared Spectroscopy Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Infrared spectroscopy enables rapid, non destructive analysis of chemical and material properties, yet high dimensional signals and overlapping bands hinder conventional chemometric methods. Large language models (LLMs), with strong generalization and reasoning capabilities, offer new opportunities for automated spectral interpretation, but their potential in this domain remains largely untapped. This study introduces LUMIR (LLM-driven Unified agent framework for Multi-task Infrared spectroscopy Reasoning), an agent based framework designed to achieve accurate infrared spectral analysis under low data conditions. LUMIR integrates a structured literature knowledge base, automated preprocessing, feature extraction, and predictive modeling into a unified pipeline. By mining peer reviewed spectroscopy studies, it identifies validated preprocessing and feature derivation strategies, transforms spectra into low dimensional representations, and applies few-shot prompts for classification, regression, and anomaly detection. The framework was validated on diverse datasets, including the publicly available Milk near-infrared dataset, Chinese medicinal herbs, Citri Reticulatae Pericarpium(CRP) with different storage durations, an industrial wastewater COD dataset, and two additional public benchmarks, Tecator and Corn. Across these tasks, LUMIR achieved performance comparable to or surpassing established machine learning and deep learning models, particularly in resource limited settings. This work demonstrates that combining structured literature guidance with few-shot learning enables robust, scalable, and automated spectral interpretation. LUMIR establishes a new paradigm for applying LLMs to infrared spectroscopy, offering high accuracy with minimal labeled data and broad applicability across scientific and industrial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17966v2">LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 arXiv admin note: substantial text overlap with arXiv:2504.15085
    </div>
    <details class="paper-abstract">
      Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02518v2">AnalogCoder-Pro: Unifying Analog Circuit Generation and Optimization via Multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      Despite recent advances, analog front-end design still relies heavily on expert intuition and iterative simulations, which limits the potential for automation. We present AnalogCoder-Pro, a multimodal large language model (LLM) framework that integrates generative and optimization techniques. The framework features a multimodal diagnosis-and-repair feedback loop that uses simulation error messages and waveform images to autonomously correct design errors. It also builds a reusable circuit tool library by archiving successful designs as modular subcircuits, accelerating the development of complex systems. Furthermore, it enables end-to-end automation by generating circuit topologies from target specifications, extracting key parameters, and applying Bayesian optimization for device sizing. On a curated benchmark suite covering 13 circuit types, AnalogCoder-Pro successfully designed 28 circuits and consistently outperformed existing LLM-based methods in figures of merit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v3">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-08-31
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11649v3">Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation</a></div>
    <div class="paper-meta">
      📅 2025-08-31
    </div>
    <details class="paper-abstract">
      We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v2">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge (LLMaaJ) now underpins scalable evaluation, yet we lack a decisive test of a judge's qualification: can it recover a conversation's latent objective and know when that inference is trustworthy? LLMs degrade under irrelevant or long context; multi-turn jailbreaks further hide goals across turns. We introduce ObjexMT, a benchmark for objective extraction and metacognition. Given a multi-turn transcript, a model must return a one-sentence base objective and a self-reported confidence. Accuracy is computed via LLM-judge semantic similarity to gold objectives, converted to binary correctness by a single human-aligned threshold calibrated once on N = 100 items ($\tau^*=0.61$). Metacognition is evaluated with ECE, Brier, Wrong-at-High-Conf, and risk-coverage. Across gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMTData_Attack600, SafeMTData_1K, MHJ, and CoSafe, claude-sonnet-4 attains the best objective-extraction accuracy (0.515) and calibration (ECE 0.296; Brier 0.324); gpt-4.1 and Qwen3-235B-A22B-FP8 tie at 0.441 but are overconfident (mean confidence $\approx$0.88 vs. accuracy $\approx$0.44; Wrong-at-0.90 $\approx$48-52%). Performance varies by dataset ($\approx$0.167-0.865). ObjexMT thus supplies an actionable test for LLM judges: when objectives are not explicit, judges often misinfer them with high confidence. We recommend exposing objectives when feasible and gating decisions by confidence otherwise. Code and data at https://github.com/hyunjun1121/ObjexMT_dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09318v3">Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLMs can generate human-like dialogues, yet their ability to simulate early child-adult interactions remains largely unexplored. In this paper, we examined how effectively LLMs can capture the distinctive features of child-caregiver language in interaction, using both static and interactive benchmarking methods. We found that state-of-the-art LLMs like Llama 3 and GPT-4o can approximate child-caregiver dialogues at the word and utterance level, but they struggle to reproduce the child and caregiver's discursive patterns, exaggerate alignment, and fail to reach the level of diversity shown by humans. The broader goal of this work is to initiate the development of a comprehensive benchmark for LLMs in child-oriented applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16534v2">Multilingual != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accepted at OMMM@RANLP2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly capable across global languages. However, the ability to communicate across languages does not necessarily translate to appropriate cultural representations. A key concern is US-centric bias, where LLMs reflect US rather than local cultural values. We propose a novel methodology that compares LLM-generated response distributions against population-level opinion data from the World Value Survey across four languages (Danish, Dutch, English, and Portuguese). Using a rigorous linear mixed-effects regression framework, we compare two families of models: Google's Gemma models (2B--27B parameters) and successive iterations of OpenAI's turbo-series. Across the families of models, we find no consistent relationships between language capabilities and cultural alignment. While the Gemma models have a positive correlation between language capability and cultural alignment across languages, the OpenAI models do not. Importantly, we find that self-consistency is a stronger predictor of multicultural alignment than multilingual capabilities. Our results demonstrate that achieving meaningful cultural alignment requires dedicated effort beyond improving general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16807v3">Detecting LLM-assisted writing in scientific communication: Are we there yet?</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), exemplified by ChatGPT, have significantly reshaped text generation, particularly in the realm of writing assistance. While ethical considerations underscore the importance of transparently acknowledging LLM use, especially in scientific communication, genuine acknowledgment remains infrequent. A potential avenue to encourage accurate acknowledging of LLM-assisted writing involves employing automated detectors. Our evaluation of four cutting-edge LLM-generated text detectors reveals their suboptimal performance compared to a simple ad-hoc detector designed to identify abrupt writing style changes around the time of LLM proliferation. We contend that the development of specialized detectors exclusively dedicated to LLM-assisted writing detection is necessary. Such detectors could play a crucial role in fostering more authentic recognition of LLM involvement in scientific communication, addressing the current challenges in acknowledgment practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04648v2">FlairGPT: Repurposing LLMs for Interior Designs</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 EUROGRAPHICS 2025
    </div>
    <details class="paper-abstract">
      Interior design involves the careful selection and arrangement of objects to create an aesthetically pleasing, functional, and harmonized space that aligns with the client's design brief. This task is particularly challenging, as a successful design must not only incorporate all the necessary objects in a cohesive style, but also ensure they are arranged in a way that maximizes accessibility, while adhering to a variety of affordability and usage considerations. Data-driven solutions have been proposed, but these are typically room- or domain-specific and lack explainability in their design design considerations used in producing the final layout. In this paper, we investigate if large language models (LLMs) can be directly utilized for interior design. While we find that LLMs are not yet capable of generating complete layouts, they can be effectively leveraged in a structured manner, inspired by the workflow of interior designers. By systematically probing LLMs, we can reliably generate a list of objects along with relevant constraints that guide their placement. We translate this information into a design layout graph, which is then solved using an off-the-shelf constrained optimization setup to generate the final layouts. We benchmark our algorithm in various design configurations against existing LLM-based methods and human designs, and evaluate the results using a variety of quantitative and qualitative metrics along with user studies. In summary, we demonstrate that LLMs, when used in a structured manner, can effectively generate diverse high-quality layouts, making them a viable solution for creating large-scale virtual scenes. Project webpage at https://flairgpt.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14533v2">Whose LLM is it Anyway? Linguistic Comparison and LLM Attribution for GPT-3.5, GPT-4 and Bard</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of generating text that is similar to or surpasses human quality. However, it is unclear whether LLMs tend to exhibit distinctive linguistic styles akin to how human authors do. Through a comprehensive linguistic analysis, we compare the vocabulary, Part-Of-Speech (POS) distribution, dependency distribution, and sentiment of texts generated by three of the most popular LLMS today (GPT-3.5, GPT-4, and Bard) to diverse inputs. The results point to significant linguistic variations which, in turn, enable us to attribute a given text to its LLM origin with a favorable 88\% accuracy using a simple off-the-shelf classification model. Theoretical and practical implications of this intriguing finding are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07227v3">LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accepted by ICCAD'25
    </div>
    <details class="paper-abstract">
      LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) techniques have been explored at the algorithmic and hardware levels. However, speculative inference results in more compute-intensive GEMM operations, creating new design trade-offs for existing GEMV-accelerated PIM architectures. Furthermore, there exists a significant amount of redundant draft tokens in tree-based speculative inference, necessitating efficient token management schemes to minimize energy consumption. In this work, we present LP-Spec, an architecture-dataflow co-design leveraging hybrid LPDDR5 performance-enhanced PIM architecture with draft token pruning and dynamic workload scheduling to accelerate LLM speculative inference. A near-data memory controller is proposed to enable data reallocation between DRAM and PIM banks. Furthermore, a data allocation unit based on the hardware-aware draft token pruner is developed to minimize energy consumption and fully exploit parallel execution opportunities. Compared to end-to-end LLM inference on other mobile solutions such as mobile NPUs or GEMV-accelerated PIMs, our LP-Spec achieves 13.21x, 7.56x, and 99.87x improvements in performance, energy efficiency, and energy-delay-product (EDP). Compared with prior AttAcc PIM and RTX 3090 GPU, LP-Spec can obtain 12.83x and 415.31x EDP reduction benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11703v2">Progent: Programmable Privilege Control for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-30
    </div>
    <details class="paper-abstract">
      LLM agents utilize Large Language Models as central components with diverse tools to complete various user tasks, but face significant security risks when interacting with external environments. Attackers can exploit these agents through various vectors, including indirect prompt injection, memory/knowledge base poisoning, and malicious tools, tricking agents into performing dangerous actions such as unauthorized financial transactions or data leakage. The core problem that enables attacks to succeed lies in over-privileged tool access. We introduce Progent, the first privilege control framework to secure LLM agents. Progent enforces security at the tool level by restricting agents to performing tool calls necessary for user tasks while blocking potentially malicious ones. Progent features a domain-specific language that allows for expressing fine-grained policies for controlling tool privileges, flexible fallback actions when calls are blocked, and dynamic policy updates to adapt to changing agent states. The framework operates deterministically at runtime, providing provable security guarantees. Thanks to our modular design, integrating Progent does not alter agent internals and only requires minimal changes to the existing agent implementation, enhancing its practicality and potential for widespread adoption. Our extensive evaluation across various agent use cases, using benchmarks like AgentDojo, ASB, and AgentPoison, demonstrates that Progent reduces attack success rates to 0%, while preserving agent utility and speed. Additionally, we show that LLMs can automatically generate effective policies, highlighting their potential for automating the process of writing Progent's security policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19720v3">Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 emnlp 2025
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12299v2">QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Accept to ACLW 2025 (WOAH)
    </div>
    <details class="paper-abstract">
      The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v4">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07268v2">Advancing Grounded Multimodal Named Entity Recognition via LLM-Based Reformulation and Box-Based Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Extension of our Findings of EMNLP 2023 & ACL 2024 paper, IEEE Transactions on Multimedia accepted on July 19, 2025
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) task aims to identify named entities, entity types and their corresponding visual regions. GMNER task exhibits two challenging attributes: 1) The tenuous correlation between images and text on social media contributes to a notable proportion of named entities being ungroundable. 2) There exists a distinction between coarse-grained noun phrases used in similar tasks (e.g., phrase localization) and fine-grained named entities. In this paper, we propose RiVEG, a unified framework that reformulates GMNER into a joint MNER-VE-VG task by leveraging large language models (LLMs) as connecting bridges. This reformulation brings two benefits: 1) It enables us to optimize the MNER module for optimal MNER performance and eliminates the need to pre-extract region features using object detection methods, thus naturally addressing the two major limitations of existing GMNER methods. 2) The introduction of Entity Expansion Expression module and Visual Entailment (VE) module unifies Visual Grounding (VG) and Entity Grounding (EG). This endows the proposed framework with unlimited data and model scalability. Furthermore, to address the potential ambiguity stemming from the coarse-grained bounding box output in GMNER, we further construct the new Segmented Multimodal Named Entity Recognition (SMNER) task and corresponding Twitter-SMNER dataset aimed at generating fine-grained segmentation masks, and experimentally demonstrate the feasibility and effectiveness of using box prompt-based Segment Anything Model (SAM) to empower any GMNER model with the ability to accomplish the SMNER task. Extensive experiments demonstrate that RiVEG significantly outperforms SoTA methods on four datasets across the MNER, GMNER, and SMNER tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12225v2">Interpretation Gaps in LLM-Assisted Comprehension of Privacy Documents</a></div>
    <div class="paper-meta">
      📅 2025-08-30
      | 💬 Minor revision
    </div>
    <details class="paper-abstract">
      This article explores the gaps that can manifest when using a large language model (LLM) to obtain simplified interpretations of data practices from a complex privacy policy. We exemplify these gaps to showcase issues in accuracy, completeness, clarity and representation, while advocating for continued research to realize an LLM's true potential in revolutionizing privacy management through personal assistants and automated compliance checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00631v2">ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Instruction tuning has underscored the significant potential of large language models (LLMs) in producing more human controllable and effective outputs in various domains. In this work, we focus on the data selection problem for task-specific instruction tuning of LLMs. Prevailing methods primarily rely on the crafted similarity metrics to select training data that aligns with the test data distribution. The goal is to minimize instruction tuning loss on the test data, ultimately improving performance on the target task. However, it has been widely observed that instruction tuning loss (i.e., cross-entropy loss for next token prediction) in LLMs often fails to exhibit a monotonic relationship with actual task performance. This misalignment undermines the effectiveness of current data selection methods for task-specific instruction tuning. To address this issue, we introduce ROSE, a novel Reward-Oriented inStruction data sElection method which leverages pairwise preference loss as a reward signal to optimize data selection for task-specific instruction tuning. Specifically, ROSE adapts an influence formulation to approximate the influence of training data points relative to a few-shot preference validation set to select the most task-related training data points. Experimental results show that by selecting just 5\% of the training data using ROSE, our approach can achieve competitive results compared to fine-tuning with the full training dataset, and it surpasses other state-of-the-art data selection methods for task-specific instruction tuning. Our qualitative analysis further confirms the robust generalizability of our method across multiple benchmark datasets and diverse model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17202v2">Active Domain Knowledge Acquisition with 100-Dollar Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21803v1">Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to The 16th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB 2025)(Poster Paper)
    </div>
    <details class="paper-abstract">
      Accurate interpretation of clinical narratives is critical for patient care, but the complexity of these notes makes automation challenging. While Large Language Models (LLMs) show promise, single-model approaches can lack the robustness required for high-stakes clinical tasks. We introduce a collaborative multi-agent system (MAS) that models a clinical consultation team to address this gap. The system is tasked with identifying clinical problems by analyzing only the Subjective (S) and Objective (O) sections of SOAP notes, simulating the diagnostic reasoning process of synthesizing raw data into an assessment. A Manager agent orchestrates a dynamically assigned team of specialist agents who engage in a hierarchical, iterative debate to reach a consensus. We evaluated our MAS against a single-agent baseline on a curated dataset of 420 MIMIC-III notes. The dynamic multi-agent configuration demonstrated consistently improved performance in identifying congestive heart failure, acute kidney injury, and sepsis. Qualitative analysis of the agent debates reveals that this structure effectively surfaces and weighs conflicting evidence, though it can occasionally be susceptible to groupthink. By modeling a clinical team's reasoning process, our system offers a promising path toward more accurate, robust, and interpretable clinical decision support tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21801v1">DMGIN: How Multimodal LLMs Enhance Large Recommendation Models for Lifelong User Post-click Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Modeling user interest based on lifelong user behavior sequences is crucial for enhancing Click-Through Rate (CTR) prediction. However, long post-click behavior sequences themselves pose severe performance issues: the sheer volume of data leads to high computational costs and inefficiencies in model training and inference. Traditional methods address this by introducing two-stage approaches, but this compromises model effectiveness due to incomplete utilization of the full sequence context. More importantly, integrating multimodal embeddings into existing large recommendation models (LRM) presents significant challenges: These embeddings often exacerbate computational burdens and mismatch with LRM architectures. To address these issues and enhance the model's efficiency and accuracy, we introduce Deep Multimodal Group Interest Network (DMGIN). Given the observation that user post-click behavior sequences contain a large number of repeated items with varying behaviors and timestamps, DMGIN employs Multimodal LLMs(MLLM) for grouping to reorganize complete lifelong post-click behavior sequences more effectively, with almost no additional computational overhead, as opposed to directly introducing multimodal embeddings. To mitigate the potential information loss from grouping, we have implemented two key strategies. First, we analyze behaviors within each group using both interest statistics and intra-group transformers to capture group traits. Second, apply inter-group transformers to temporally ordered groups to capture the evolution of user group interests. Our extensive experiments on both industrial and public datasets confirm the effectiveness and efficiency of DMGIN. The A/B test in our LBS advertising system shows that DMGIN improves CTR by 4.7% and Revenue per Mile by 2.3%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v4">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo (Fine-grained Semantic Comparison), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSCo more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17196v2">BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v3">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Under review. 19 pages, 8 figures, 12 tables. Code and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning. The code (https://github.com/yyysjz1997/Time-RA) and dataset (https://huggingface.co/datasets/Time-RA/RATs40K) have been fully open-sourced to support and accelerate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v2">Robustness is Important: Limitations of LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21628v1">Personality Matters: User Traits Predict LLM Preferences in Multi-Turn Collaborative Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly integrate into everyday workflows, where users shape outcomes through multi-turn collaboration, a critical question emerges: do users with different personality traits systematically prefer certain LLMs over others? We conducted a study with 32 participants evenly distributed across four Keirsey personality types, evaluating their interactions with GPT-4 and Claude 3.5 across four collaborative tasks: data analysis, creative writing, information retrieval, and writing assistance. Results revealed significant personality-driven preferences: Rationals strongly preferred GPT-4, particularly for goal-oriented tasks, while idealists favored Claude 3.5, especially for creative and analytical tasks. Other personality types showed task-dependent preferences. Sentiment analysis of qualitative feedback confirmed these patterns. Notably, aggregate helpfulness ratings were similar across models, showing how personality-based analysis reveals LLM differences that traditional evaluations miss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v1">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted by EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21561v1">Summarize-Exemplify-Reflect: Data-driven Insight Distillation Empowers LLMs for Few-shot Tabular Classification</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 25 Findings
    </div>
    <details class="paper-abstract">
      Recent studies show the promise of large language models (LLMs) for few-shot tabular classification but highlight challenges due to the variability in structured data. To address this, we propose distilling data into actionable insights to enable robust and effective classification by LLMs. Drawing inspiration from human learning processes, we introduce InsightTab, an insight distillation framework guided by principles of divide-and-conquer, easy-first, and reflective learning. Our approach integrates rule summarization, strategic exemplification, and insight reflection through deep collaboration between LLMs and data modeling techniques. The obtained insights enable LLMs to better align their general knowledge and capabilities with the particular requirements of specific tabular tasks. We extensively evaluate InsightTab on nine datasets. The results demonstrate consistent improvement over state-of-the-art methods. Ablation studies further validate the principle-guided distillation process, while analyses emphasize InsightTab's effectiveness in leveraging labeled data and managing bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21540v1">HealthProcessAI: A Technical Framework and Proof-of-Concept for LLM-Enhanced Healthcare Process Mining</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Process mining has emerged as a powerful analytical technique for understanding complex healthcare workflows. However, its application faces significant barriers, including technical complexity, a lack of standardized approaches, and limited access to practical training resources. We introduce HealthProcessAI, a GenAI framework designed to simplify process mining applications in healthcare and epidemiology by providing a comprehensive wrapper around existing Python (PM4PY) and R (bupaR) libraries. To address unfamiliarity and improve accessibility, the framework integrates multiple Large Language Models (LLMs) for automated process map interpretation and report generation, helping translate technical analyses into outputs that diverse users can readily understand. We validated the framework using sepsis progression data as a proof-of-concept example and compared the outputs of five state-of-the-art LLM models through the OpenRouter platform. To test its functionality, the framework successfully processed sepsis data across four proof-of-concept scenarios, demonstrating robust technical performance and its capability to generate reports through automated LLM analysis. LLM evaluation using five independent LLMs as automated evaluators revealed distinct model strengths: Claude Sonnet-4 and Gemini 2.5-Pro achieved the highest consistency scores (3.79/4.0 and 3.65/4.0) when evaluated by automated LLM assessors. By integrating multiple Large Language Models (LLMs) for automated interpretation and report generation, the framework addresses widespread unfamiliarity with process mining outputs, making them more accessible to clinicians, data scientists, and researchers. This structured analytics and AI-driven interpretation combination represents a novel methodological advance in translating complex process mining results into potentially actionable insights for healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21512v1">Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17052v2">Testing Conviction: An Argumentative Framework for Measuring LLM Political Stability</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape political discourse, yet exhibit inconsistent responses when challenged. While prior research categorizes LLMs as left- or right-leaning based on single-prompt responses, a critical question remains: Do these classifications reflect stable ideologies or superficial mimicry? Existing methods cannot distinguish between genuine ideological alignment and performative text generation. To address this, we propose a framework for evaluating ideological depth through (1) argumentative consistency and (2) uncertainty quantification. Testing 12 LLMs on 19 economic policies from the Political Compass Test, we classify responses as stable or performative ideological positioning. Results show 95% of left-leaning models and 89% of right-leaning models demonstrate behavior consistent with our classifications across different experimental conditions. Furthermore, semantic entropy strongly validates our classifications (AUROC=0.78), revealing uncertainty's relationship to ideological consistency. Our findings demonstrate that ideological stability is topic-dependent and challenge the notion of monolithic LLM ideologies, and offer a robust way to distinguish genuine alignment from performative behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21476v1">Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at https://github.com/weixiaolong94-hub/Igniting-Creative-Writing-in-Small-Language-Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20863v2">Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21452v1">From Canonical to Complex: Benchmarking LLM Capabilities in Undergraduate Thermodynamics</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Benchmark downloadable at https://huggingface.co/datasets/herteltm/UTQA
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly considered as tutoring aids in science education. Yet their readiness for unsupervised use in undergraduate instruction remains uncertain, as reliable teaching requires more than fluent recall: it demands consistent, principle-grounded reasoning. Thermodynamics, with its compact laws and subtle distinctions between state and path functions, reversibility, and entropy, provides an ideal testbed for evaluating such capabilities. Here we present UTQA, a 50-item undergraduate thermodynamics question answering benchmark, covering ideal-gas processes, reversibility, and diagram interpretation. No leading 2025-era model exceeded our 95\% competence threshold: the best LLMs achieved 82\% accuracy, with text-only items performing better than image reasoning tasks, which often fell to chance levels. Prompt phrasing and syntactic complexity showed modest to little correlation with performance. The gap concentrates in finite-rate/irreversible scenarios and in binding visual features to thermodynamic meaning, indicating that current LLMs are not yet suitable for unsupervised tutoring in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v1">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21417v1">An Empirical Study of Vulnerable Package Dependencies in LLM Repositories</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have developed rapidly in recent years, revolutionizing various fields. Despite their widespread success, LLMs heavily rely on external code dependencies from package management systems, creating a complex and interconnected LLM dependency supply chain. Vulnerabilities in dependencies can expose LLMs to security risks. While existing research predominantly focuses on model-level security threats, vulnerabilities within the LLM dependency supply chain have been overlooked. To fill this gap, we conducted an empirical analysis of 52 open-source LLMs, examining their third-party dependencies and associated vulnerabilities. We then explored activities within the LLM repositories to understand how maintainers manage third-party vulnerabilities in practice. Finally, we compared third-party dependency vulnerabilities in the LLM ecosystem to those in the Python ecosystem. Our results show that half of the vulnerabilities in the LLM ecosystem remain undisclosed for more than 56.2 months, significantly longer than those in the Python ecosystem. Additionally, 75.8% of LLMs include vulnerable dependencies in their configuration files. This study advances the understanding of LLM supply chain risks, provides insights for practitioners, and highlights potential directions for improving the security of the LLM supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v3">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21378v1">RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17178v3">SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/zjukg/SKA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v5">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21323v1">LLM-driven Provenance Forensics for Threat Investigation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      We introduce PROVSEEK, an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction. PROVSEEK employs specialized toolchains to dynamically retrieve relevant context by generating precise, context-aware queries that fuse a vectorized threat report knowledge base with data from system provenance databases. The framework resolves provenance queries, orchestrates multiple role-specific agents to mitigate hallucinations, and synthesizes structured, ground-truth verifiable forensic summaries. By combining agent orchestration with Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning, PROVSEEK enables adaptive multi-step analysis that iteratively refines hypotheses, verifies supporting evidence, and produces scalable, interpretable forensic explanations of attack behaviors. By combining provenance data with agentic reasoning, PROVSEEK establishes a new paradigm for grounded agentic forecics to investigate APTs. We conduct a comprehensive evaluation on publicly available DARPA datasets, demonstrating that PROVSEEK outperforms retrieval-based methods for intelligence extraction task, achieving a 34% improvement in contextual precision/recall; and for threat detection task, PROVSEEK achieves 22%/29% higher precision/recall compared to both a baseline agentic AI approach and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection System (PIDS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21305v1">LLM-Supported Content Analysis of Motivated Reasoning on Climate Change</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 11 pages, 3 figures. Accepted for ASIS&T 2025
    </div>
    <details class="paper-abstract">
      Public discourse around climate change remains polarized despite scientific consensus on anthropogenic climate change (ACC). This study examines how "believers" and "skeptics" of ACC differ in their YouTube comment discourse. We analyzed 44,989 comments from 30 videos using a large language model (LLM) as a qualitative annotator, identifying ten distinct topics. These annotations were combined with social network analysis to examine engagement patterns. A linear mixed-effects model showed that comments about government policy and natural cycles generated significantly lower interaction compared to misinformation, suggesting these topics are ideologically settled points within communities. These patterns reflect motivated reasoning, where users selectively engage with content that aligns with their identity and beliefs. Our findings highlight the utility of LLMs for large-scale qualitative analysis and highlight how climate discourse is shaped not only by content, but by underlying cognitive and ideological motivations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21300v1">Improving Fisher Information Estimation and Efficiency for LoRA-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated remarkable performance across various tasks but face challenges related to unintentionally generating outputs containing sensitive information. A straightforward approach to address this issue is to retrain the model after excluding the problematic data. However, this approach incurs prohibitively high computational costs. To overcome this limitation, machine unlearning has emerged as a promising solution that can effectively remove sensitive information without the need to retrain the model from scratch. Recently, FILA has been proposed as a parameter-efficient unlearning method by integrating LoRA adapters. Specifically, it calculates the Fisher information to identify parameters associated with the forget set and assigns them to LoRA adapters for updates. Despite its innovative approach, FILA still requires access to all model parameters and does not adequately account for fundamental assumptions underlying Fisher information, leading to inaccuracies in importance estimation. To address these limitations, we propose VILA, a novel unlearning framework that explicitly considers the assumptions overlooked in FILA, thereby enhancing the accuracy of parameter identification for the forget set. Moreover, VILA significantly reduces computational costs by enabling parameter identification without accessing the entire model. Our method achieves up to 100x higher parameter efficiency and 40x faster training speed compared to FILA, and sets new state-of-the-art performance on benchmarks including TOFU, WMDP, and MUSE. Our code is available at https://github.com/kyj93790/VILA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21285v1">A Financial Brain Scan of the LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 47 pages
    </div>
    <details class="paper-abstract">
      Emerging techniques in computer science make it possible to "brain scan" large language models (LLMs), identify the plain-English concepts that guide their reasoning, and steer them while holding other factors constant. We show that this approach can map LLM-generated economic forecasts to concepts such as sentiment, technical analysis, and timing, and compute their relative importance without reducing performance. We also show that models can be steered to be more or less risk-averse, optimistic, or pessimistic, which allows researchers to correct or simulate biases. The method is transparent, lightweight, and replicable for empirical research in the social sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v4">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in list{\ttfamily .}get(i). Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21273v1">CALM: A Framework for Continuous, Adaptive, and LLM-Mediated Anomaly Detection in Time-Series Streams</a></div>
    <div class="paper-meta">
      📅 2025-08-29
    </div>
    <details class="paper-abstract">
      The detection of anomalies in non-stationary time-series streams is a critical but challenging task across numerous industrial and scientific domains. Traditional models, trained offline, suffer significant performance degradation when faced with concept drift, where the underlying statistical properties of the data change over time. This paper introduces CALM (Continuous, Adaptive, and LLM-Mediated), a novel, end-to-end framework for real-time anomaly detection designed to address this challenge. CALM is built on the Apache Beam distributed processing framework and leverages the TimesFm foundation model for forecasting-based anomaly detection. The framework's novelty lies in two core contributions. First, it implements a closed-loop, continuous fine-tuning mechanism that allows the anomaly detection model to adapt to evolving data patterns in near real-time. Second, it introduces an LLM-as-a-Judge component, a Large Language Model that provides semantic, context-aware judgments on detected anomalies to curate a high-quality training dataset, deciding whether an anomaly represents transient noise or a meaningful pattern shift. We evaluate CALM on the comprehensive TSB-UAD benchmark. Our results demonstrate that the continuously fine-tuned model improves the ROC AUC score in most datasets compared to the static, pre-trained base model, validating the efficacy of our adaptive, LLM-guided approach to maintaining high-performance anomaly detection in dynamic streaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00273v2">Echoes in AI: Quantifying lack of plot diversity in LLM outputs</a></div>
    <div class="paper-meta">
      📅 2025-08-29
      | 💬 PNAS Vol. 122 No. 35. Copyright \c{opyright} 2025 the Author(s). Published by PNAS. This open access article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND)
    </div>
    <details class="paper-abstract">
      With rapid advances in large language models (LLMs), there has been an increasing application of LLMs in creative content ideation and generation. A critical question emerges: can current LLMs provide ideas that are diverse enough to truly bolster collective creativity? We examine two state-of-the-art LLMs, GPT-4 and LLaMA-3, on story generation and discover that LLM-generated stories often consist of plot elements that are echoed across a number of generations. To quantify this phenomenon, we introduce the Sui Generis score, an automatic metric that measures the uniqueness of a plot element among alternative storylines generated using the same prompt under an LLM. Evaluating on 100 short stories, we find that LLM-generated stories often contain combinations of idiosyncratic plot elements echoed frequently across generations and across different LLMs, while plots from the original human-written stories are rarely recreated or even echoed in pieces. Moreover, our human evaluation shows that the ranking of Sui Generis scores among story segments correlates moderately with human judgment of surprise level, even though score computation is completely automatic without relying on human judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14862v2">Bitune: Leveraging Bidirectional Attention to Improve Decoder-Only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Decoder-only large language models typically rely solely on masked causal attention, which limits their expressiveness by restricting information flow to one direction. We propose Bitune, a method that enhances pretrained decoder-only LLMs by incorporating bidirectional attention into prompt processing. We evaluate Bitune in instruction-tuning and question-answering settings, showing significant improvements in performance on commonsense reasoning, arithmetic, and language understanding tasks. Furthermore, extensive ablation studies validate the role of each component of the method, and demonstrate that Bitune is compatible with various parameter-efficient finetuning techniques and full model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20996v1">ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20912v1">Research Challenges in Relational Database Management Systems for LLM Queries</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 This paper will appear in the 6th International Workshop on Applied AI for Database Systems and Applications, AIDB Workshop at VLDB 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential for applications such as text summarization, sentiment analysis, and automated question-answering. Recently, LLMs have also been integrated into relational database management systems to enhance querying and support advanced data processing. Companies such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly within SQL, denoted as LLM queries, to boost data insights. However, open-source solutions currently have limited functionality and poor performance. In this work, we present an early exploration of two open-source systems and one enterprise platform, using five representative queries to expose functional, performance, and scalability limits in today's SQL-invoked LLM integrations. We identify three main issues: enforcing structured outputs, optimizing resource utilization, and improving query planning. We implemented initial solutions and observed improvements in accommodating LLM powered SQL queries. These early gains demonstrate that tighter integration of LLM+DBMS is the key to scalable and efficient processing of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20863v1">Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20818v1">cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 A shorter version has been accepted to the 2025 Conference on Information and Knowledge Management
    </div>
    <details class="paper-abstract">
      Many multi-agent reinforcement learning (MARL) algorithms are trained in fixed simulation environments, making them brittle when deployed in real-world scenarios with more complex and uncertain conditions. Contextual MARL (cMARL) addresses this by parameterizing environments with context variables and training a context-agnostic policy that performs well across all environment configurations. Existing cMARL methods attempt to use curriculum learning to help train and evaluate context-agnostic policies, but they often rely on unreliable proxy signals, such as value estimates or generalized advantage estimates that are noisy and unstable in multi-agent settings due to inter-agent dynamics and partial observability. To address these issues, we propose Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending (cMALC-D), a framework that uses Large Language Models (LLMs) to generate semantically meaningful curricula and provide a more robust evaluation signal. To prevent mode collapse and encourage exploration, we introduce a novel diversity-based context blending mechanism that creates new training scenarios by combining features from prior contexts. Experiments in traffic signal control domains demonstrate that cMALC-D significantly improves both generalization and sample efficiency compared to existing curriculum learning baselines. We provide code at https://github.com/DaRL-LibSignal/cMALC-D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20810v1">A Graph-Based Test-Harness for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 4 pages, 2 figures, dataset
    </div>
    <details class="paper-abstract">
      We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at https://github.com/jessicalundin/graph_testing_harness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08846v2">Steering Towards Fairness: Mitigating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at CASE@RANLP2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases along political and economic dimensions. In this paper, we employ a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), this method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v3">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [Accepted]. To be held on 26th of October, 2025
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20764v1">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025,14 page,3 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we conduct the first comparative analysis of emotional arcs between real and LLM-generated Cognitive Behavioral Therapy dialogues. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions transcribed from public videos and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability,more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity between real and synthetic speakers is low, especially for clients. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. We introduce RealCBT, a curated dataset of real CBT sessions, to support future research in this space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20750v1">Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection across Datasets</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Paper accepted at the DHOW Workshop at ACM Multimedia 2025. Code available at https://github.com/idiap/implicit-hsd
    </div>
    <details class="paper-abstract">
      Implicit hate speech (IHS) is indirect language that conveys prejudice or hatred through subtle cues, sarcasm or coded terminology. IHS is challenging to detect as it does not include explicit derogatory or inflammatory words. To address this challenge, task-specific pipelines can be complemented with external knowledge or additional information such as context, emotions and sentiment data. In this paper, we show that, by solely fine-tuning recent general-purpose embedding models based on large language models (LLMs), such as Stella, Jasper, NV-Embed and E5, we achieve state-of-the-art performance. Experiments on multiple IHS datasets show up to 1.10 percentage points improvements for in-dataset, and up to 20.35 percentage points improvements in cross-dataset evaluation, in terms of F1-macro score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20744v1">From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Context: Laws and regulations increasingly affect software design and quality assurance, but legal texts are written in technology-neutral language. This creates challenges for engineers who must develop compliance artifacts such as requirements and acceptance criteria. Manual creation is labor-intensive, error-prone, and requires domain expertise. Advances in Generative AI (GenAI), especially Large Language Models (LLMs), offer a way to automate deriving such artifacts. Objective: We present the first systematic human-subject study of LLMs' ability to derive behavioral specifications from legal texts using a quasi-experimental design. These specifications translate legal requirements into a developer-friendly form. Methods: Ten participants evaluated specifications generated from food-safety regulations by Claude and Llama. Using Gherkin, a structured BDD language, 60 specifications were produced. Each participant assessed 12 across five criteria: Relevance, Clarity, Completeness, Singularity, and Time Savings. Each specification was reviewed by two participants, yielding 120 assessments. Results: For Relevance, 75% of ratings were highest and 20% second-highest. Clarity reached 90% highest. Completeness: 75% highest, 19% second. Singularity: 82% highest, 12% second. Time Savings: 68% highest, 24% second. No lowest ratings occurred. Mann-Whitney U tests showed no significant differences across participants or models. Llama slightly outperformed Claude in Clarity, Completeness, and Time Savings, while Claude was stronger in Singularity. Feedback noted hallucinations and omissions but confirmed the utility of the specifications. Conclusion: LLMs can generate high-quality Gherkin specifications from legal texts, reducing manual effort and providing structured artifacts useful for implementation, assurance, and test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20737v1">Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18321v2">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: https://github.com/declare-lab/KAIROS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20697v1">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20643v1">CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Code: https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19720v2">Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03661v2">Automated Algorithmic Discovery for Gravitational-Wave Detection Guided by LLM-Informed Evolutionary Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 79 pages (29 main), with 6+6 figures and 2 tables, presenting a more concise and updated manuscript
    </div>
    <details class="paper-abstract">
      Gravitational-wave signal detection with unknown source parameters buried in dynamic detector noise remains a formidable computational challenge. Existing approaches face core limitations from restrictive assumptions: traditional methods rely on predefined theoretical priors, while neural networks introduce hidden biases and lack interpretability. We propose Evolutionary Monte Carlo Tree Search (Evo-MCTS), the first integration of large language model (LLM) guidance with domain-aware physical constraints for automated gravitational wave detection. This framework systematically explores algorithmic solution spaces through tree-structured search enhanced by evolutionary optimization, combining MCTS for strategic exploration with evolutionary algorithms for solution refinement. The LLM component provides domain-aware heuristics while maintaining interpretability through explicit algorithmic pathway generation. Experimental validation demonstrates substantial performance improvements, achieving a 20.2% improvement over state-of-the-art gravitational wave detection algorithms on the MLGWSC-1 benchmark dataset and a remarkable 59.1% improvement over other LLM-based algorithm optimization frameworks. Beyond performance improvements, our framework establishes a transferable methodology for automated algorithmic discovery across computational science domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15165v2">SoAy: A Solution-based LLM API-using Methodology for Academic Information Seeking</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 KDD 2025; 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) for academic API usage shows promise in reducing researchers' academic information seeking efforts. However, current LLM API-using methods struggle with complex API coupling commonly encountered in academic queries. To address this, we introduce SoAy, a solution-based LLM API-using methodology for academic information seeking. It uses code with a solution as the reasoning method, where a solution is a pre-constructed API calling sequence. The addition of the solution reduces the difficulty for the model to understand the complex relationships between APIs. Code improves the efficiency of reasoning. To evaluate SoAy, we introduce SoAyBench, an evaluation benchmark accompanied by SoAyEval, built upon a cloned environment of APIs from AMiner. Experimental results demonstrate a 34.58-75.99\% performance improvement compared to state-of-the-art LLM API-based baselines. All datasets, codes, tuned models, and deployed online services are publicly accessible at https://github.com/RUCKBReasoning/SoAy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17387v2">Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20525v1">Enhancing Health Fact-Checking with LLM-Generated Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Fact-checking for health-related content is challenging due to the limited availability of annotated training data. In this study, we propose a synthetic data generation pipeline that leverages large language models (LLMs) to augment training data for health-related fact checking. In this pipeline, we summarize source documents, decompose the summaries into atomic facts, and use an LLM to construct sentence-fact entailment tables. From the entailment relations in the table, we further generate synthetic text-claim pairs with binary veracity labels. These synthetic data are then combined with the original data to fine-tune a BERT-based fact-checking model. Evaluation on two public datasets, PubHealth and SciFact, shows that our pipeline improved F1 scores by up to 0.019 and 0.049, respectively, compared to models trained only on the original data. These results highlight the effectiveness of LLM-driven synthetic data augmentation in enhancing the performance of health-related fact-checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v4">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20514v1">SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06056v2">Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16201v2">SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens to enable efficient speculation without sacrificing accuracy. To achieve this, we performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B. Code is available at https://github.com/zju-jiyicheng/SpecVLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20462v1">Automated Quality Assessment for LLM-Based Complex Qualitative Coding: A Confidence-Diversity Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 21 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      While previous research demonstrated effective automated quality assessment for accessible LLM coding tasks, a fundamental question remains: can confidence-diversity frameworks maintain reliability for complex analytical tasks requiring specialized domain expertise and extensive text comprehension? Traditional inter-coder reliability measures become prohibitively expensive at scale, yet the lack of reliable automated quality assessment methods creates methodological barriers to AI adoption in sophisticated qualitative research. This study extends dual-signal quality assessment combining model confidence and inter-model consensus from accessible to complex analytical domains. We systematically validate this approach across three domains: legal reasoning (390 Supreme Court cases), political analysis (645 hyperpartisan articles), and medical classification (1,000 clinical transcripts). Results demonstrate that uncertainty-based indicators maintain predictive validity in complex tasks, with external entropy showing consistent negative correlations with accuracy (r = -0.179 to -0.273, p < 0.001) and confidence exhibiting positive correlations in two domains (r = 0.104 to 0.429). Systematic weight optimization achieves 6.6 to 113.7 percent improvements over single-signal approaches, with optimized weights transferring effectively across domains (100 percent success rate). An intelligent triage system reduces manual verification effort by 44.6 percent while maintaining quality standards. These findings establish that automated quality assessment can scale from accessible to complex analytical tasks, providing practical tools for expanding AI-assisted qualitative research. Future work will focus on addressing long-tail challenges in high-disagreement, low-confidence cases to further enhance screening efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20453v1">MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We introduce MCP-Bench, a benchmark for evaluating large language models (LLMs) on realistic, multi-step tasks that demand tool use, cross-tool coordination, precise parameter control, and planning/reasoning for solving tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28 representative live MCP servers spanning 250 tools across domains such as finance, traveling, scientific computing, and academic search. Unlike prior API-based benchmarks, each MCP server provides a set of complementary tools designed to work together, enabling the construction of authentic, multi-step tasks with rich input-output coupling. Tasks in MCP-Bench test agents' ability to retrieve relevant tools from fuzzy instructions without explicit tool names, plan multi-hop execution trajectories for complex objectives, ground responses in intermediate tool outputs, and orchestrate cross-domain workflows - capabilities not adequately evaluated by existing benchmarks that rely on explicit tool specifications, shallow few-step workflows, and isolated domain operations. We propose a multi-faceted evaluation framework covering tool-level schema understanding and usage, trajectory-level planning, and task completion. Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench. Code and data: https://github.com/Accenture/mcp-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20444v1">Ransomware 3.0: Self-Composing and LLM-Orchestrated</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20443v1">Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Aware Unlearning with Proxy Constraint</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods lack a sound forgetting boundary, causing some samples to be under-forgotten, leaving residual leakage risks, while others remain over-forgotten at the expense of degraded utility. In this work, we propose EAGLE-PC (Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint), a novel unlearning framework that addresses these limitations through two key components. First, entanglement-awareness guided loss reweighting determines the forgetting effort of each sample by measuring its similarity to retain samples in the embedding space, enabling more targeted and effective unlearning. Second, a proxy constraint leveraging ICL (In-Context Learning) generated test data softly regularizes the forgetting process, effectively mitigating over-forgetting. EAGLE-PC is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EAGLE-PC on the TOFU and MUSE benchmarks, showing consistent improvements in the forgetting-utility trade-off across multiple LLMs. Combined with the NPO+GD optimizer, it approaches full retraining performance, offering a scalable and robust unlearning solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20420v1">CAMB: A comprehensive industrial LLM benchmark on civil aviation maintenance</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Civil aviation maintenance is a domain characterized by stringent industry standards. Within this field, maintenance procedures and troubleshooting represent critical, knowledge-intensive tasks that require sophisticated reasoning. To address the lack of specialized evaluation tools for large language models (LLMs) in this vertical, we propose and develop an industrial-grade benchmark specifically designed for civil aviation maintenance. This benchmark serves a dual purpose: It provides a standardized tool to measure LLM capabilities within civil aviation maintenance, identifying specific gaps in domain knowledge and complex reasoning. By pinpointing these deficiencies, the benchmark establishes a foundation for targeted improvement efforts (e.g., domain-specific fine-tuning, RAG optimization, or specialized prompt engineering), ultimately facilitating progress toward more intelligent solutions within civil aviation maintenance. Our work addresses a significant gap in the current LLM evaluation, which primarily focuses on mathematical and coding reasoning tasks. In addition, given that Retrieval-Augmented Generation (RAG) systems are currently the dominant solutions in practical applications , we leverage this benchmark to evaluate existing well-known vector embedding models and LLMs for civil aviation maintenance scenarios. Through experimental exploration and analysis, we demonstrate the effectiveness of our benchmark in assessing model performance within this domain, and we open-source this evaluation benchmark and code to foster further research and development:https://github.com/CamBenchmark/cambenchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20416v1">DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20401v1">Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 In Proceedings of 2nd Workshop on Evaluating and Applying Recommendation Systems with Large Language Models (EARL) at RecSys 2025 (EARL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for recommendation tasks due to their general-purpose capabilities. While LLMs perform well in rich-context settings, their behavior in cold-start scenarios, where only limited signals such as age, gender, or language are available, raises fairness concerns because they may rely on societal biases encoded during pretraining. We introduce a benchmark specifically designed to evaluate fairness in zero-context recommendation. Our modular pipeline supports configurable recommendation domains and sensitive attributes, enabling systematic and flexible audits of any open-source LLM. Through evaluations of state-of-the-art models (Gemma 3 and Llama 3.2), we uncover consistent biases across recommendation domains (music, movies, and colleges) including gendered and cultural stereotypes. We also reveal a non-linear relationship between model size and fairness, highlighting the need for nuanced analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20395v1">Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision. We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20384v1">Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Under review for AAAI 2026
    </div>
    <details class="paper-abstract">
      In this work, we introduce Entropy Area Score (EAS), a simple yet effective metric to quantify uncertainty in the answer generation process of reasoning large language models (LLMs). EAS requires neither external models nor repeated sampling, it integrates token-level predictive entropy from the model itself to capture the evolution of uncertainty during generation. Empirical results show that EAS is strongly correlated with answer entropy across models and datasets. In training data selection, EAS identifies high-potential samples and consistently outperforms Pass Rate filtering under equal sample budgets, improving student model accuracy on math benchmarks. EAS is both efficient and interpretable, offering a practical tool for uncertainty modeling and data quality assessment in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v3">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in $\texttt{list.get(i)}$. Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v2">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20373v1">Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at https://github.com/Graph-Reasoner/Graph-R1, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20351v1">Joint Enhancement of Relational Reasoning for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 9 pages, 5 pages Accepted by EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite significant progress, large language models (LLMs) still struggle with long contexts due to memory limitations and their inability to tackle complex and long-context tasks. Additionally, LLMs often suffer from a lack of transparency and are prone to producing hallucinations. To address these challenges, we propose \textbf{JERR}, a novel framework designed to enhance long-context comprehension via graph-based reasoning in LLMs. JERR integrates three key components: synopsis extraction, graph construction, and relational reasoning. First, synopsis is extracted by chunking text strategically, allowing the model to summarize and understand information more efficiently. Second, we build a directed acyclic graph (DAG) to resolve redundancy, ensuring logical consistency and clarity. Finally, we incorporate Monte Carlo Tree Search (MCTS) to help the model navigate complex reasoning paths, ensuring more accurate and interpretable outputs. This framework provides a novel solution that enables LLMs to handle extended contexts and complex reasoning tasks with improved reliability and transparency. Experimental results show that JERR consistently outperforms all baselines on the ROUGE and F1 metrics, achieving the highest scores on the LLM-Rater evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20340v1">Boosting Skeleton-Driven SMT Solver Fuzzing by Leveraging LLM to Produce Formula Generators</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with the LLMs introduce substantial computational overhead. In this study, we present Chimera, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of reusable term (i.e., logical expression) generators. Particularly, Chimera uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Chimera populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Chimera requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Chimera on two leading SMT solvers: Z3 and cvc5. Our experiments show that Chimera has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19512v3">Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks often leads to catastrophic forgetting, notably degrading the safety of originally aligned models. While some existing methods attempt to restore safety by incorporating additional safety data, the quality of such data typically falls short of that used in the original alignment process. Moreover, these high-quality safety datasets are generally inaccessible, making it difficult to fully recover the model's original safety. We ask: How can we preserve safety while improving downstream task performance without additional safety data? We show that simply merging the weights of pre- and post-fine-tuned models effectively mitigates safety degradation while enhancing performance. Experiments across different downstream tasks and models validate the method's practicality and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v3">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20333v1">Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.
    </details>
</div>
