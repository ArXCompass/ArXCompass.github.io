# llm - 2025_09

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20321v1">DRES: Benchmarking LLMs for Disfluency Removal</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09873v2">LLM-Powered AI Tutors with Personas for d/Deaf and Hard-of-Hearing Online Learners</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems (ITS) using artificial intelligence (AI) technology have shown promise in supporting learners with diverse abilities. Large language models (LLMs) provide new opportunities to incorporate personas to AI-based tutors and support dynamic interactive dialogue. This paper explores how DHH learners interact with LLM-powered AI tutors with different experiences in DHH education as personas to identify their accessibility preferences. A user study with 16 DHH participants showed that they asked DHH-related questions based on background information and evaluated the AI tutors' cultural knowledge of the DHH communities in their responses. Participants suggested providing more transparency in each AI tutor's position within the DHH community. Participants also pointed out the lack of support in the multimodality of sign language in current LLMs. We discuss design implications to support the diverse needs in interaction between DHH users and the LLMs, such as offering supports in tuning language styles of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v1">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at https://anonymous.4open.science/r/judgment-to-noise-947D/README.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20278v1">Instruction Boundary: Quantifying Biases in LLM Reasoning under Various Coverage</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) reasoning has long been regarded as a powerful tool for problem solving across domains, providing non-experts with valuable advice. However, their limitations - especially those stemming from prompt design - remain underexplored. Because users may supply biased or incomplete prompts - often unintentionally - LLMs can be misled, undermining reliability and creating risks. We refer to this vulnerability as the Instruction Boundary. To investigate the phenomenon, we distill it into eight concrete facets and introduce BiasDetector, a framework that measures biases arising from three instruction types: complete, redundant, and insufficient. We evaluate several mainstream LLMs and find that, despite high headline accuracy, substantial biases persist in many downstream tasks as a direct consequence of prompt coverage. Our empirical study confirms that LLM reasoning reliability can still be significantly improved. We analyze the practical impact of these biases and outline mitigation strategies. Our findings underscore the need for developers to tackle biases and for users to craft options carefully.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14264v2">AAPO: Enhancing the Reasoning Capabilities of LLMs with Advantage Momentum</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a momentum-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03547v2">Guided Reality: Generating Visually-Enriched AR Task Guidance with LLMs and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 To appear at UIST 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the automatic generation of step-by-step augmented reality (AR) instructions for a wide range of physical tasks. However, existing LLM-based AR guidance often lacks rich visual augmentations to effectively embed instructions into spatial context for a better user understanding. We present Guided Reality, a fully automated AR system that generates embedded and dynamic visual guidance based on step-by-step instructions. Our system integrates LLMs and vision models to: 1) generate multi-step instructions from user queries, 2) identify appropriate types of visual guidance, 3) extract spatial information about key interaction points in the real world, and 4) embed visual guidance in physical space to support task execution. Drawing from a corpus of user manuals, we define five categories of visual guidance and propose an identification strategy based on the current step. We evaluate the system through a user study (N=16), completing real-world tasks and exploring the system in the wild. Additionally, four instructors shared insights on how Guided Reality could be integrated into their training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18920v2">Context-Masked Meta-Prompting for Privacy-Preserving LLM Adaptation in Finance</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      The increasing reliance on Large Language Models (LLMs) in sensitive domains like finance necessitates robust methods for privacy preservation and regulatory compliance. This paper presents an iterative meta-prompting methodology designed to optimise hard prompts without exposing proprietary or confidential context to the LLM. Through a novel regeneration process involving feeder and propagation methods, we demonstrate significant improvements in prompt efficacy. Evaluated on public datasets serving as proxies for financial tasks such as SQuAD for extractive financial Q&A, CNN/DailyMail for news summarisation, and SAMSum for client interaction summarisation, our approach, utilising GPT-3.5 Turbo, achieved a 103.87% improvement in ROUGE-L F1 for question answering. This work highlights a practical, low-cost strategy for adapting LLMs to financial applications while upholding critical privacy and auditability standards, offering a compelling case for its relevance in the evolving landscape of generative AI in finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20230v1">Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20214v1">Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at https://github.com/snu-mllab/Q-Palette.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20208v1">Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at https://github.com/parkervg/blendsql
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18058v2">Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are crafted to be subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using them as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19269v1">Extracting Conceptual Spaces from LLMs Using Prototype Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Conceptual spaces represent entities and concepts using cognitively meaningful dimensions, typically referring to perceptual features. Such representations are widely used in cognitive science and have the potential to serve as a cornerstone for explainable AI. Unfortunately, they have proven notoriously difficult to learn, although recent LLMs appear to capture the required perceptual features to a remarkable extent. Nonetheless, practical methods for extracting the corresponding conceptual spaces are currently still lacking. While various methods exist for extracting embeddings from LLMs, extracting conceptual spaces also requires us to encode the underlying features. In this paper, we propose a strategy in which features (e.g. sweetness) are encoded by embedding the description of a corresponding prototype (e.g. a very sweet food). To improve this strategy, we fine-tune the LLM to align the prototype embeddings with the corresponding conceptual space dimensions. Our empirical analysis finds this approach to be highly effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15433v2">LLM-Driven SAST-Genius: A Hybrid Static Analysis Framework for Comprehensive and Actionable Security</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      This report examines the synergy between Large Language Models (LLMs) and Static Application Security Testing (SAST) to improve vulnerability discovery. Traditional SAST tools, while effective for proactive security, are limited by high false-positive rates and a lack of contextual understanding. Conversely, LLMs excel at code analysis and pattern recognition but can be prone to inconsistencies and hallucinations. By integrating these two technologies, a more intelligent and efficient system is created. This combination moves beyond mere vulnerability detection optimization, transforming security into a deeply integrated, contextual process that provides tangible benefits like improved triage, dynamic bug descriptions, bug validation via exploit generation and enhanced analysis of complex codebases. The result is a more effective security approach that leverages the strengths of both technologies while mitigating their weaknesses. SAST-Genius reduced false positives by about 91 % (225 to 20) compared to Semgrep alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19265v1">Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 EMNLP 2025 - Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often reflect Western-centric biases, limiting their effectiveness in diverse cultural contexts. Although some work has explored cultural alignment, the potential for cross-cultural transfer, using alignment in one culture to improve performance in others, remains underexplored. This paper investigates cross-cultural transfer of commonsense reasoning in the Arab world, where linguistic and historical similarities coexist with local cultural differences. Using a culturally grounded commonsense reasoning dataset covering 13 Arab countries, we evaluate lightweight alignment methods such as in-context learning and demonstration-based reinforcement (DITTO), alongside baselines like supervised fine-tuning and direct preference optimization. Our results show that merely 12 culture-specific examples from one country can improve performance in others by 10\% on average, within multilingual models. In addition, we demonstrate that out-of-culture demonstrations from Indonesia and US contexts can match or surpass in-culture alignment for MCQ reasoning, highlighting cultural commonsense transferability beyond the Arab world. These findings demonstrate that efficient cross-cultural alignment is possible and offer a promising approach to adapt LLMs to low-resource cultural settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08727v3">Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 ICCV 2025, Project page: https://boyangdeng.com/visual-chronicles , second and third listed authors have equal contributions
    </div>
    <details class="paper-abstract">
      We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at https://boyangdeng.com/visual-chronicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11593v2">Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 This manuscript has been accepted for publication in Springer Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19153v1">LLMs as verification oracles for Solidity</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19136v1">On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11189v2">Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can amplify misinformation, undermining societal goals like the UN SDGs. We study three documented drivers of misinformation (valence framing, information overload, and oversimplification) which are often shaped by one's default beliefs. Building on evidence that LLMs encode such defaults (e.g., "joy is positive," "math is complex") and can act as "bags of heuristics," we ask: can general belief-driven heuristics behind misinformative behaviour be recovered from LLMs as clear rules? A key obstacle is that global rule-extraction methods in explainable AI (XAI) are built for numerical inputs/outputs, not text. We address this by eliciting global LLM beliefs and mapping them to numerical scores via statistically reliable abstractions, thereby enabling off-the-shelf global XAI to detect belief-related heuristics in LLMs. To obtain ground truth, we hard-code bias-inducing nonlinear heuristics of increasing complexity (univariate, conjunctive, nonconvex) into popular LLMs (ChatGPT and Llama) via system instructions. This way, we find that RuleFit under-detects non-univariate biases, while global SHAP better approximates conjunctive ones but does not yield actionable rules. To bridge this gap, we propose RuleSHAP, a rule-extraction algorithm that couples global SHAP-value aggregations with rule induction to better capture non-univariate bias, improving heuristics detection over RuleFit by +94% (MRR@1) on average. Our results provide a practical pathway for revealing belief-driven biases in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17310v4">Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Guesstimation--the task of making approximate quantitative estimates about objects or events-is a common real--world skill, yet remains underexplored in large language model (LLM) research. We introduce three guesstimation datasets: MARBLES, FUTURE, and ELECPRED, spanning physical estimation (e.g., how many marbles fit in a cup) to abstract predictions (e.g., the 2024 U.S. presidential election). Inspired by the social science concept of Wisdom of Crowds (WOC)- where the median of multiple estimates improves accuracy-we propose WOC decoding for LLMs. We replicate WOC effects in human participants and find that LLMs exhibit similar benefits: median aggregation across sampled responses consistently improves accuracy over greedy decoding, self-consistency decoding, and mean decoding. This suggests that LLMs encode a world model that supports approximate reasoning. Our results position guesstimation as a useful probe of LLM world knowledge and highlight WOC decoding as a strategy for enhancing LLM guesstimation performance on real-world tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19125v1">Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted to EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      The rapid growth of scientific literature demands efficient methods to organize and synthesize research findings. Existing taxonomy construction methods, leveraging unsupervised clustering or direct prompting of large language models (LLMs), often lack coherence and granularity. We propose a novel context-aware hierarchical taxonomy generation framework that integrates LLM-guided multi-aspect encoding with dynamic clustering. Our method leverages LLMs to identify key aspects of each paper (e.g., methodology, dataset, evaluation) and generates aspect-specific paper summaries, which are then encoded and clustered along each aspect to form a coherent hierarchy. In addition, we introduce a new evaluation benchmark of 156 expert-crafted taxonomies encompassing 11.6k papers, providing the first naturally annotated dataset for this task. Experimental results demonstrate that our method significantly outperforms prior approaches, achieving state-of-the-art performance in taxonomy coherence, granularity, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19117v1">LLM-based Vulnerability Discovery through the Lens of Code Metrics</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many tasks of software engineering, yet progress in leveraging them for vulnerability discovery has stalled in recent years. To understand this phenomenon, we investigate LLMs through the lens of classic code metrics. Surprisingly, we find that a classifier trained solely on these metrics performs on par with state-of-the-art LLMs for vulnerability discovery. A root-cause analysis reveals a strong correlation and a causal effect between LLMs and code metrics: When the value of a metric is changed, LLM predictions tend to shift by a corresponding magnitude. This dependency suggests that LLMs operate at a similarly shallow level as code metrics, limiting their ability to grasp complex patterns and fully realize their potential in vulnerability discovery. Based on these findings, we derive recommendations on how research should more effectively address this challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19104v1">DRO-REBEL: Distributionally Robust Relative-Reward Regression for Fast and Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 70 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Reinforcement learning with human feedback (RLHF) has become crucial for aligning Large Language Models (LLMs) with human intent. However, existing offline RLHF approaches suffer from overoptimization, where models overfit to reward misspecification and drift from preferred behaviors observed during training. We introduce DRO-REBEL, a unified family of robust REBEL updates with type-$p$ Wasserstein, KL, and $\chi^2$ ambiguity sets. Using Fenchel duality, each update reduces to a simple relative-reward regression, preserving scalability and avoiding PPO-style clipping or auxiliary value networks. Under standard linear-reward and log-linear policy classes with a data-coverage condition, we establish $O(n^{-1/4})$ estimation bounds with tighter constants than prior DRO-DPO approaches, and recover the minimax-optimal $O(n^{-1/2})$ rate via a localized Rademacher complexity analysis. The same analysis closes the gap for Wasserstein-DPO and KL-DPO, showing both also attain optimal parametric rates. We derive practical SGD algorithms for all three divergences: gradient regularization (Wasserstein), importance weighting (KL), and a fast 1-D dual solve ($\chi^2$). Experiments on Emotion Alignment, the large-scale ArmoRM multi-objective benchmark, and HH-Alignment demonstrate strong worst-case robustness across unseen preference mixtures, model sizes, and data scales, with $\chi^2$-REBEL showing consistently strong empirical performance. A controlled radius--coverage study validates a no-free-lunch trade-off: radii shrinking faster than empirical divergence concentration rates achieve minimax-optimal parametric rates but forfeit coverage, while coverage-guaranteeing radii incur $O(n^{-1/4})$ rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19057v1">RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Biomedical knowledge graphs (KGs) are vital for drug discovery and clinical decision support but remain incomplete. Large language models (LLMs) excel at extracting biomedical relations, yet their outputs lack standardization and alignment with ontologies, limiting KG integration. We introduce RELATE, a three-stage pipeline that maps LLM-extracted relations to standardized ontology predicates using ChemProt and the Biolink Model. The pipeline includes: (1) ontology preprocessing with predicate embeddings, (2) similarity-based retrieval enhanced with SapBERT, and (3) LLM-based reranking with explicit negation handling. This approach transforms relation extraction from free-text outputs to structured, ontology-constrained representations. On the ChemProt benchmark, RELATE achieves 52% exact match and 94% accuracy@10, and in 2,400 HEAL Project abstracts, it effectively rejects irrelevant associations (0.4%) and identifies negated assertions. RELATE captures nuanced biomedical relationships while ensuring quality for KG augmentation. By combining vector search with contextual LLM reasoning, RELATE provides a scalable, semantically accurate framework for converting unstructured biomedical literature into standardized KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18555v2">Unraveling Misinformation Propagation in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted to EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning, positioning them as promising tools for supporting human problem-solving. However, what happens when their performance is affected by misinformation, i.e., incorrect inputs introduced by users due to oversights or gaps in knowledge? Such misinformation is prevalent in real-world interactions with LLMs, yet how it propagates within LLMs' reasoning process remains underexplored. Focusing on mathematical reasoning, we present a comprehensive analysis of how misinformation affects intermediate reasoning steps and final answers. We also examine how effectively LLMs can correct misinformation when explicitly instructed to do so. Even with explicit instructions, LLMs succeed less than half the time in rectifying misinformation, despite possessing correct internal knowledge, leading to significant accuracy drops (10.02% - 72.20%), and the degradation holds with thinking models (4.30% - 19.97%). Further analysis shows that applying factual corrections early in the reasoning process most effectively reduces misinformation propagation, and fine-tuning on synthesized data with early-stage corrections significantly improves reasoning factuality. Our work offers a practical approach to mitigating misinformation propagation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11079v2">Difficulty-Aware Agent Orchestration in LLM-Powered Workflows</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or task-level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailoring workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13978v2">LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Paper accepted in the proceedings of the Supercomputing Conference (SC). Cite it as Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, and Rafael Ferreira da Silva. LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology. In WORKS at the ACM/IEEE International Conference on Supercomputing, 2025
    </div>
    <details class="paper-abstract">
      Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18980v1">From latent factors to language: a user study on LLM-generated explanations for an inherently interpretable matrix-based recommender system</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      We investigate whether large language models (LLMs) can generate effective, user-facing explanations from a mathematically interpretable recommendation model. The model is based on constrained matrix factorization, where user types are explicitly represented and predicted item scores share the same scale as observed ratings, making the model's internal representations and predicted scores directly interpretable. This structure is translated into natural language explanations using carefully designed LLM prompts. Many works in explainable AI rely on automatic evaluation metrics, which often fail to capture users' actual needs and perceptions. In contrast, we adopt a user-centered approach: we conduct a study with 326 participants who assessed the quality of the explanations across five key dimensions-transparency, effectiveness, persuasion, trust, and satisfaction-as well as the recommendations themselves.To evaluate how different explanation strategies are perceived, we generate multiple explanation types from the same underlying model, varying the input information provided to the LLM. Our analysis reveals that all explanation types are generally well received, with moderate statistical differences between strategies. User comments further underscore how participants react to each type of explanation, offering complementary insights beyond the quantitative results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18970v1">LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18965v1">Benchmarking PDF Accessibility Evaluation A Dataset and Framework for Assessing Automated and LLM-Based Approaches for Accessibility Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      PDFs remain the dominant format for scholarly communication, despite significant accessibility challenges for blind and low-vision users. While various tools attempt to evaluate PDF accessibility, there is no standardized methodology to evaluate how different accessibility assessment approaches perform. Our work addresses this critical gap by introducing a novel benchmark dataset of scholarly PDFs with expert-validated accessibility annotations across seven criteria (alternative text quality, logical reading order, semantic tagging, table structure, functional hyperlinks, color contrast, and font readability), and a four-category evaluation framework with standardized labels (Passed, Failed, Not Present, Cannot Tell) to systematically assess accessibility evaluation approaches. Using our evaluation framework, we explore whether large language models (LLMs) are capable of supporting automated accessibility evaluation. We benchmark five LLMs, which demonstrate varying capabilities in correctly assessing different accessibility criteria, with GPT-4-Turbo achieving the highest overall accuracy (0.85). However, all models struggled in correctly categorizing documents with Not Present and Cannot Tell accessibility labels, particularly for alt text quality assessment. Our qualitative comparison with standard automated checkers reveals complementary strengths: rule-based tools excel at technical verification, while LLMs better evaluate semantic appropriateness and contextual relevance. Based on our findings, we propose a hybrid approach that would combine automated checkers, LLM evaluation, and human assessment as a future strategy for PDF accessibility evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17998v2">Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted as Poster at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The efficiency of Bayesian optimization (BO) relies heavily on the choice of the Gaussian process (GP) kernel, which plays a central role in balancing exploration and exploitation under limited evaluation budgets. Traditional BO methods often rely on fixed or heuristic kernel selection strategies, which can result in slow convergence or suboptimal solutions when the chosen kernel is poorly suited to the underlying objective function. To address this limitation, we propose a freshly-baked Context-Aware Kernel Evolution (CAKE) to enhance BO with large language models (LLMs). Concretely, CAKE leverages LLMs as the crossover and mutation operators to adaptively generate and refine GP kernels based on the observed data throughout the optimization process. To maximize the power of CAKE, we further propose BIC-Acquisition Kernel Ranking (BAKER) to select the most effective kernel through balancing the model fit measured by the Bayesian information criterion (BIC) with the expected improvement at each iteration of BO. Extensive experiments demonstrate that our fresh CAKE-based BO method consistently outperforms established baselines across a range of real-world tasks, including hyperparameter optimization, controller tuning, and photonic chip design. Our code is publicly available at https://github.com/richardcsuwandi/cake.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18934v1">Generic Adversarial Smart Contract Detection with Semantics and Uncertainty-Aware LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Adversarial smart contracts, mostly on EVM-compatible chains like Ethereum and BSC, are deployed as EVM bytecode to exploit vulnerable smart contracts typically for financial gains. Detecting such malicious contracts at the time of deployment is an important proactive strategy preventing loss from victim contracts. It offers a better cost-benefit than detecting vulnerabilities on diverse potential victims. However, existing works are not generic with limited detection types and effectiveness due to imbalanced samples, while the emerging LLM technologies, which show its potentials in generalization, have two key problems impeding its application in this task: hard digestion of compiled-code inputs, especially those with task-specific logic, and hard assessment of LLMs' certainty in their binary answers, i.e., yes-or-no answers. Therefore, we propose a generic adversarial smart contracts detection framework FinDet, which leverages LLMs with two enhancements addressing above two problems. FinDet takes as input only the EVM-bytecode contracts and identifies adversarial ones among them with high balanced accuracy. The first enhancement extracts concise semantic intentions and high-level behavioral logic from the low-level bytecode inputs, unleashing the LLM reasoning capability restricted by the task input. The second enhancement probes and measures the LLM uncertainty to its multi-round answering to the same query, improving the LLM answering robustness for binary classifications required by the task output. Our comprehensive evaluation shows that FinDet achieves a BAC of 0.9223 and a TPR of 0.8950, significantly outperforming existing baselines. It remains robust under challenging conditions including unseen attack patterns, low-data settings, and feature obfuscation. FinDet detects all 5 public and 20+ unreported adversarial contracts in a 10-day real-world test, confirmed manually.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15335v2">PolBiX: Detecting LLMs' Political Bias in Fact-Checking through X-phemisms</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted at Findings of EMNLP 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly used in applications requiring objective assessment, which could be compromised by political bias. Many studies found preferences for left-leaning positions in LLMs, but downstream effects on tasks like fact-checking remain underexplored. In this study, we systematically investigate political bias through exchanging words with euphemisms or dysphemisms in German claims. We construct minimal pairs of factually equivalent claims that differ in political connotation, to assess the consistency of LLMs in classifying them as true or false. We evaluate six LLMs and find that, more than political leaning, the presence of judgmental words significantly influences truthfulness assessment. While a few models show tendencies of political bias, this is not mitigated by explicitly calling for objectivism in prompts. Warning: This paper contains content that may be offensive or upsetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18886v1">Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed on converged Cloud and High-Performance Computing (HPC) infrastructure. However, as LLMs handle confidential inputs and are fine-tuned on costly, proprietary datasets, their heightened security requirements slow adoption in privacy-sensitive sectors such as healthcare and finance. We investigate methods to address this gap and propose Trusted Execution Environments (TEEs) as a solution for securing end-to-end LLM inference. We validate their practicality by evaluating these compute-intensive workloads entirely within CPU and GPU TEEs. On the CPU side, we conduct an in-depth study running full Llama2 inference pipelines (7B, 13B, 70B) inside Intel's TDX and SGX, accelerated by Advanced Matrix Extensions (AMX). We derive 12 insights, including that across various data types, batch sizes, and input lengths, CPU TEEs impose under 10% throughput and 20% latency overheads, further reduced by AMX. We run LLM inference on NVIDIA H100 Confidential Compute GPUs, contextualizing our CPU findings and observing throughput penalties of 4-8% that diminish as batch and input sizes grow. By comparing performance, cost, and security trade-offs, we show how CPU TEEs can be more cost-effective or secure than their GPU counterparts. To our knowledge, our work is the first to comprehensively demonstrate the performance and practicality of modern TEEs across both CPUs and GPUs for enabling confidential LLMs (cLLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18874v1">When Ads Become Profiles: Large-Scale Audit of Algorithmic Biases and LLM Profiling Risks</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Automated ad targeting on social media is opaque, creating risks of exploitation and invisibility to external scrutiny. Users may be steered toward harmful content while independent auditing of these processes remains blocked. Large Language Models (LLMs) raise a new concern: the potential to reverse-engineer sensitive user attributes from exposure alone. We introduce a multi-stage auditing framework to investigate these risks. First, a large-scale audit of over 435,000 ad impressions delivered to 891 Australian Facebook users reveals algorithmic biases, including disproportionate Gambling and Politics ads shown to socioeconomically vulnerable and politically aligned groups. Second, a multimodal LLM can reconstruct users' demographic profiles from ad streams, outperforming census-based baselines and matching or exceeding human performance. Our results provide the first empirical evidence that ad streams constitute rich digital footprints for public AI inference, highlighting urgent privacy risks and the need for content-level auditing and governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09790v2">Prompting for Performance: Exploring LLMs for Configuring Software</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 ICTAI 2025
    </div>
    <details class="paper-abstract">
      Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18846v1">Model selection meets clinical semantics: Optimizing ICD-10-CM prediction via LLM-as-Judge evaluation, redundancy-aware sampling, and section-aware fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 28 Pages, 4 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Accurate International Classification of Diseases (ICD) coding is critical for clinical documentation, billing, and healthcare analytics, yet it remains a labour-intensive and error-prone task. Although large language models (LLMs) show promise in automating ICD coding, their challenges in base model selection, input contextualization, and training data redundancy limit their effectiveness. We propose a modular framework for ICD-10 Clinical Modification (ICD-10-CM) code prediction that addresses these challenges through principled model selection, redundancy-aware data sampling, and structured input design. The framework integrates an LLM-as-judge evaluation protocol with Plackett-Luce aggregation to assess and rank open-source LLMs based on their intrinsic comprehension of ICD-10-CM code definitions. We introduced embedding-based similarity measures, a redundancy-aware sampling strategy to remove semantically duplicated discharge summaries. We leverage structured discharge summaries from Taiwanese hospitals to evaluate contextual effects and examine section-wise content inclusion under universal and section-specific modelling paradigms. Experiments across two institutional datasets demonstrate that the selected base model after fine-tuning consistently outperforms baseline LLMs in internal and external evaluations. Incorporating more clinical sections consistently improves prediction performance. This study uses open-source LLMs to establish a practical and principled approach to ICD-10-CM code prediction. The proposed framework provides a scalable, institution-ready solution for real-world deployment of automated medical coding systems by combining informed model selection, efficient data refinement, and context-aware prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15826v2">Campus AI vs. Commercial AI: How Customizations Shape Trust and Usage of LLM as-a-Service Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Added missing author to the author list; no other changes
    </div>
    <details class="paper-abstract">
      As the use of LLM chatbots by students and researchers becomes more prevalent, universities are pressed to develop AI strategies. One strategy that many universities pursue is to customize pre-trained LLM as-a-service (LLMaaS). While most studies on LLMaaS chatbots prioritize technical adaptations, we focus on psychological effects of user-salient customizations, such as interface changes. We assume that such customizations influence users' perception of the system and are therefore important in guiding safe and appropriate use. In a field study, we examine how students and employees (N = 526) at a German university perceive and use their institution's customized LLMaaS chatbot compared to ChatGPT. Participants using both systems (n = 116) reported greater trust, higher perceived privacy and less experienced hallucinations with their university's customized LLMaaS chatbot in contrast to ChatGPT. We discuss theoretical implications for research on calibrated trust, and offer guidance on the design and deployment of LLMaaS chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18843v1">Are Smaller Open-Weight LLMs Closing the Gap to Proprietary Models for Biomedical Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 CLEF 2025 Working Notes, 9-12 September 2025, Madrid, Spain
    </div>
    <details class="paper-abstract">
      Open-weight versions of large language models (LLMs) are rapidly advancing, with state-of-the-art models like DeepSeek-V3 now performing comparably to proprietary LLMs. This progression raises the question of whether small open-weight LLMs are capable of effectively replacing larger closed-source models. We are particularly interested in the context of biomedical question-answering, a domain we explored by participating in Task 13B Phase B of the BioASQ challenge. In this work, we compare several open-weight models against top-performing systems such as GPT-4o, GPT-4.1, Claude 3.5 Sonnet, and Claude 3.7 Sonnet. To enhance question answering capabilities, we use various techniques including retrieving the most relevant snippets based on embedding distance, in-context learning, and structured outputs. For certain submissions, we utilize ensemble approaches to leverage the diverse outputs generated by different models for exact-answer questions. Our results demonstrate that open-weight LLMs are comparable to proprietary ones. In some instances, open-weight LLMs even surpassed their closed counterparts, particularly when ensembling strategies were applied. All code is publicly available at https://github.com/evidenceprime/BioASQ-13b.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15561v2">Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18808v1">SR-Eval: Evaluating LLMs on Code Generation under Stepwise Requirement Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in code generation. However, existing benchmarks mainly formalize the task as a static, single-turn problem, overlooking the stepwise requirement changes and iterative workflows in real-world software development. This mismatch limits the understanding of how well LLMs can support real-world development workflows. Constructing such iterative benchmarks is challenging due to the lack of public interaction traces and the difficulty of creating discriminative, turn-specific test cases. To bridge this gap, we present SR-Eval, a benchmark specifically designed to assess LLMs on iterative code generation under Stepwise requirements Refinement. SR-Eval spans both function-level and repository-level tasks in Python and Java, enabling fine-grained and progressive evaluation across evolving requirements. The construction of SR-Eval follows a carefully designed pipeline that first leverages a multi-agent-based requirement generation method to simulate the development process and recover the multi-round interaction process from final requirements, then employs a semantic-aware discriminative test case generation component to ensure discriminative and consistent evaluation at each turn. SR-Eval comprises 443 multi-turn tasks and 1,857 questions at both function and repository levels. Using SR-Eval, we evaluate 11 representative LLMs with three prompting strategies that simulate different usage patterns. Results show that iterative code generation under stepwise requirement refinement remains highly challenging: the best-performing model achieves only 22.67% completion rate on function-level tasks and 20.00% on repository-level tasks. We further observe that prompting strategies substantially influence performance, highlighting the need for the development of advanced methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v3">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08378v2">Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed on mobile devices, but the limited DRAM capacity constrains the deployable model size. This paper introduces ActiveFlow, the first LLM inference framework that can achieve adaptive DRAM usage for modern LLMs (not ReLU-based), enabling the scaling up of deployable model sizes. The framework is based on the novel concept of active weight DRAM-flash swapping and incorporates three novel techniques: (1) Cross-layer active weights preloading. It uses the activations from the current layer to predict the active weights of several subsequent layers, enabling computation and data loading to overlap, as well as facilitating large I/O transfers. (2) Sparsity-aware self-distillation. It adjusts the active weights to align with the dense-model output distribution, compensating for approximations introduced by contextual sparsity. (3) Active weight DRAM-flash swapping pipeline. It orchestrates the DRAM space allocation among the hot weight cache, preloaded active weights, and computation-involved weights based on available memory. Results show ActiveFlow achieves the performance-cost Pareto frontier compared to existing efficiency optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15260v2">Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 9 pages, EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16216v2">Memorization or Reasoning? Exploring the Idiom Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19892v2">OptMerge: Unifying Multimodal LLM Capabilities and Modalities via Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Foundation models update slowly due to resource-intensive training, whereas domain-specific models evolve rapidly between releases. Model merging seeks to combine multiple expert models into a single, more capable model, reducing storage and serving costs while supporting decentralized development. Despite its potential, previous studies have primarily focused on merging visual classification models or Large Language Models (LLMs) for code and math tasks. Recently, Multimodal LLMs (MLLMs) that extend LLMs through large-scale multimodal training have gained traction. However, there lacks a benchmark for model merging research that clearly divides the tasks for MLLM training and evaluation. In this paper, $\textbf{(i)}$ we introduce a model merging benchmark for MLLMs, which includes multiple tasks such as VQA, Geometry, Chart, OCR, and Grounding, studying both LoRA and full fine-tuning models. Moreover, we explore how model merging can combine different modalities (e.g., vision-language, audio-language, and video-language models), moving toward the Omni-language model. $\textbf{(ii)}$ We implement 10 model merging algorithms on the benchmark. Furthermore, we propose a novel method that removes noise from task vectors and robustly optimizes the merged vector based on a loss defined over task vector interactions, achieving an average performance gain of 2.48%. $\textbf{(iii)}$ We find that model merging offers a promising way for building improved MLLMs without requiring training data. Our results also demonstrate that the complementarity among multiple modalities outperforms individual modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18719v1">LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 12 pages, 12 figures, ACL 2025 industry track
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to e-commerce payment fraud detection by integrating reinforcement learning (RL) with Large Language Models (LLMs). By framing transaction risk as a multi-step Markov Decision Process (MDP), RL optimizes risk detection across multiple payment stages. Crafting effective reward functions, essential for RL model success, typically requires significant human expertise due to the complexity and variability in design. LLMs, with their advanced reasoning and coding capabilities, are well-suited to refine these functions, offering improvements over traditional methods. Our approach leverages LLMs to iteratively enhance reward functions, achieving better fraud detection accuracy and demonstrating zero-shot capability. Experiments with real-world data confirm the effectiveness, robustness, and resilience of our LLM-enhanced RL framework through long-term evaluations, underscoring the potential of LLMs in advancing industrial RL applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v3">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18700v1">Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Music Information Retrieval (MIR) encompasses a broad range of computational techniques for analyzing and understanding musical content, with recent deep learning advances driving substantial improvements. Building upon these advances, this paper explores how large language models (LLMs) can serve as an integrative bridge to connect and integrate information from multiple MIR tools, with a focus on enhancing automatic chord recognition performance. We present a novel approach that positions text-based LLMs as intelligent coordinators that process and integrate outputs from diverse state-of-the-art MIR tools-including music source separation, key detection, chord recognition, and beat tracking. Our method converts audio-derived musical information into textual representations, enabling LLMs to perform reasoning and correction specifically for chord recognition tasks. We design a 5-stage chain-of-thought framework that allows GPT-4o to systematically analyze, compare, and refine chord recognition results by leveraging music-theoretical knowledge to integrate information across different MIR components. Experimental evaluation on three datasets demonstrates consistent improvements across multiple evaluation metrics, with overall accuracy gains of 1-2.77% on the MIREX metric. Our findings demonstrate that LLMs can effectively function as integrative bridges in MIR pipelines, opening new directions for multi-tool coordination in music information retrieval tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13694v2">StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted by MICRO'25
    </div>
    <details class="paper-abstract">
      Efficient execution of deep learning workloads on dataflow architectures is crucial for overcoming memory bottlenecks and maximizing performance. While streaming intermediate results between computation kernels can significantly improve efficiency, existing approaches struggle with inter-kernel correlations, external memory access management, and buffer optimization. In this work, we propose StreamTensor, a compiler framework that automatically constructs and optimizes stream-based dataflow accelerators. StreamTensor introduces a novel iterative tensor type system to explicitly encode stream layouts, enabling seamless kernel fusion, buffer allocation, and memory optimization. By systematically exploring three hierarchical design spaces, including tensor tiling, kernel fusion, and resource allocation, StreamTensor balances computational intensity, memory efficiency, and data streaming to maximize performance. Based on FPGA evaluations on Large Language Models (LLM), StreamTensor achieves up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs, making it a promising approach for scalable dataflow-based deep learning acceleration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18156v2">Can LLMs Explain Themselves Counterfactually?</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18661v1">Agentic AutoSurvey: Let LLMs Survey LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 29 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The exponential growth of scientific literature poses unprecedented challenges for researchers attempting to synthesize knowledge across rapidly evolving fields. We present \textbf{Agentic AutoSurvey}, a multi-agent framework for automated survey generation that addresses fundamental limitations in existing approaches. Our system employs four specialized agents (Paper Search Specialist, Topic Mining \& Clustering, Academic Survey Writer, and Quality Evaluator) working in concert to generate comprehensive literature surveys with superior synthesis quality. Through experiments on six representative LLM research topics from COLM 2024 categories, we demonstrate that our multi-agent approach achieves significant improvements over existing baselines, scoring 8.18/10 compared to AutoSurvey's 4.77/10. The multi-agent architecture processes 75--443 papers per topic (847 total across six topics) while targeting high citation coverage (often $\geq$80\% on 75--100-paper sets; lower on very large sets such as RLHF) through specialized agent orchestration. Our 12-dimension evaluation captures organization, synthesis integration, and critical analysis beyond basic metrics. These findings demonstrate that multi-agent architectures represent a meaningful advancement for automated literature survey generation in rapidly evolving scientific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18658v1">Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 To appear in EMNLP 2025. Our code and data are available at \url{https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03293v2">LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Modified version of prior LTLCrit work with new robotics dataset
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in zero-shot and single step reasoning and decision making problems, but in long horizon sequential planning tasks, their errors compound, often leading to unreliable or inefficient behavior. We introduce LogicGuard, a modular actor-critic architecture in which an LLM actor is guided by a trajectory level LLM critic that communicates through Linear Temporal Logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. LogicGuard supports both fixed safety rules and adaptive, learned constraints, and is model-agnostic: any LLM-based planner can serve as the actor, with LogicGuard acting as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LogicGuard to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. To demonstrate generality, we evaluate LogicGuard across two distinct settings: short-horizon general tasks and long-horizon specialist tasks. On the Behavior benchmark of 100 household tasks, LogicGuard increases task completion rates by 25% over a baseline InnerMonologue planner. On the Minecraft diamond-mining task, which is long-horizon and requires multiple interdependent subgoals, LogicGuard improves both efficiency and safety compared to SayCan and InnerMonologue. These results show that enabling LLMs to supervise each other through temporal logic yields more reliable, efficient and safe decision-making for both embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19470v3">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18575v1">The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance in information retrieval tasks like passage ranking. Our research examines how instruction-following capabilities in LLMs interact with multi-document comparison tasks, identifying what we term the "Ranking Blind Spot", a characteristic of LLM decision processes during comparative evaluation. We analyze how this ranking blind spot affects LLM evaluation systems through two approaches: Decision Objective Hijacking, which alters the evaluation goal in pairwise ranking systems, and Decision Criteria Hijacking, which modifies relevance standards across ranking schemes. These approaches demonstrate how content providers could potentially influence LLM-based ranking systems to affect document positioning. These attacks aim to force the LLM ranker to prefer a specific passage and rank it at the top. Malicious content providers can exploit this weakness, which helps them gain additional exposure by attacking the ranker. In our experiment, We empirically show that the proposed attacks are effective in various LLMs and can be generalized to multiple ranking schemes. We apply these attack to realistic examples to show their effectiveness. We also found stronger LLMs are more vulnerable to these attacks. Our code is available at: https://github.com/blindspotorg/RankingBlindSpot
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18569v1">Explore the Reinforcement Learning for the LLM based ASR and TTS system</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18557v1">LLMZ+: Contextual Prompt Whitelist Principles for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 7 pages, 5 figures, to be published and presented at ICMLA 2025
    </div>
    <details class="paper-abstract">
      Compared to traditional models, agentic AI represents a highly valuable target for potential attackers as they possess privileged access to data sources and API tools, which are traditionally not incorporated into classical agents. Unlike a typical software application residing in a Demilitarized Zone (DMZ), agentic LLMs consciously rely on nondeterministic behavior of the AI (only defining a final goal, leaving the path selection to LLM). This characteristic introduces substantial security risk to both operational security and information security. Most common existing defense mechanism rely on detection of malicious intent and preventing it from reaching the LLM agent, thus protecting against jailbreak attacks such as prompt injection. In this paper, we present an alternative approach, LLMZ+, which moves beyond traditional detection-based approaches by implementing prompt whitelisting. Through this method, only contextually appropriate and safe messages are permitted to interact with the agentic LLM. By leveraging the specificity of context, LLMZ+ guarantees that all exchanges between external users and the LLM conform to predefined use cases and operational boundaries. Our approach streamlines the security framework, enhances its long-term resilience, and reduces the resources required for sustaining LLM information security. Our empirical evaluation demonstrates that LLMZ+ provides strong resilience against the most common jailbreak prompts. At the same time, legitimate business communications are not disrupted, and authorized traffic flows seamlessly between users and the agentic LLM. We measure the effectiveness of approach using false positive and false negative rates, both of which can be reduced to 0 in our experimental setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16516v2">LLM-Guided Co-Training for Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a novel weighted co-training approach that is guided by Large Language Models (LLMs). Namely, in our co-training approach, we use LLM labels on unlabeled data as target labels and co-train two encoder-only based networks that train each other over multiple iterations: first, all samples are forwarded through each network and historical estimates of each network's confidence in the LLM label are recorded; second, a dynamic importance weight is derived for each sample according to each network's belief in the quality of the LLM label for that sample; finally, the two networks exchange importance weights with each other -- each network back-propagates all samples weighted with the importance weights coming from its peer network and updates its own parameters. By strategically utilizing LLM-generated guidance, our approach significantly outperforms conventional SSL methods, particularly in settings with abundant unlabeled data. Empirical results show that it achieves state-of-the-art performance on 4 out of 5 benchmark datasets and ranks first among 14 compared methods according to the Friedman test. Our results highlight a new direction in semi-supervised learning -- where LLMs serve as knowledge amplifiers, enabling backbone co-training models to achieve state-of-the-art performance efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v4">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted at Findings of 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Meta (previously known as Facebook) advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. Additionally, we conduct a comprehensive fairness analysis to uncover biases in model predictions. We assess disparities in accuracy and error rates across demographic groups using established fairness metrics such as Demographic Parity, Equal Opportunity, and Predictive Equality. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of male audiences. The analysis of thematic explanations uncovers recurring patterns in messaging strategies tailored to various demographic groups, while the fairness analysis underscores the need for more inclusive targeting methods. This study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23817v2">MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      General matrix-vector multiplication (GeMV) remains a critical latency bottleneck in large language model (LLM) inference, even with quantized low-bit models. Processing-Using-DRAM (PUD), an analog in-DRAM computing technique, has the potential to repurpose on-device DRAM as a GeMV engine, offering additional high-throughput processing capabilities to widespread consumer devices without DRAM modifications. However, applying PUD to GeMV operations in the LLM inference pipeline incurs significant overheads $\textit{before}$ and $\textit{after}$ in-DRAM computation, diminishing the benefits of its high-throughput processing capabilities. This paper presents MVDRAM, the first practical system to accelerate GeMV operations for low-bit LLM inference using unmodified DRAM. By leveraging the data sharing patterns and mathematical linearity in GeMV operations, MVDRAM orchestrates the processor and DRAM to eliminate the costs associated with pre-arranging inputs and bit-transposition of outputs required in conventional PUD approaches. Our experimental evaluation with four DDR4 DRAM modules shows that MVDRAM achieves comparable or even better inference speed than the processor-based implementation for GeMV operations in low-bit (under 4-bit) LLM. In particular, MVDRAM achieves up to 7.29$\times$ speedup and 30.5$\times$ energy efficiency for low-bit GeMV operations. For end-to-end LLM inference, MVDRAM achieves 2.18$\times$ and 1.31$\times$ throughput improvements, along with 3.04$\times$ and 2.35$\times$ energy efficiency, for 2-bit and 4-bit quantized low-bit models, respectively. MVDRAM has the potential to redefine the AI hardware landscape by demonstrating the feasibility of standard DRAM as an LLM accelerator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18487v1">Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for text-rich graph machine learning tasks such as node classification in high-impact domains like fraud detection and recommendation systems. Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in their interaction with graph data. In this work, we conduct a large-scale, controlled evaluation across several key axes of variability to systematically assess the strengths and weaknesses of LLM-based graph reasoning methods in text-based applications. The axes include the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; structural regimes contrasting homophilic and heterophilic graphs; feature characteristics involving both short- and long-text node attributes; and model configurations with varying LLM sizes and reasoning capabilities. We further analyze dependencies by methodically truncating features, deleting edges, and removing labels to quantify reliance on input types. Our findings provide practical and actionable guidance. (1) LLMs as code generators achieve the strongest overall performance on graph data, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation is able to flexibly adapt its reliance between structure, features, or labels to leverage the most informative input type. Together, these findings provide a comprehensive view of the strengths and limitations of current LLM-graph interaction modes and highlight key design principles for future approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18400v1">ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Accurate classification of products under the Harmonized Tariff Schedule (HTS) is a critical bottleneck in global trade, yet it has received little attention from the machine learning community. Misclassification can halt shipments entirely, with major postal operators suspending deliveries to the U.S. due to incomplete customs documentation. We introduce the first benchmark for HTS code classification, derived from the U.S. Customs Rulings Online Search System (CROSS). Evaluating leading LLMs, we find that our fine-tuned Atlas model (LLaMA-3.3-70B) achieves 40 percent fully correct 10-digit classifications and 57.5 percent correct 6-digit classifications, improvements of 15 points over GPT-5-Thinking and 27.5 points over Gemini-2.5-Pro-Thinking. Beyond accuracy, Atlas is roughly five times cheaper than GPT-5-Thinking and eight times cheaper than Gemini-2.5-Pro-Thinking, and can be self-hosted to guarantee data privacy in high-stakes trade and compliance workflows. While Atlas sets a strong baseline, the benchmark remains highly challenging, with only 40 percent 10-digit accuracy. By releasing both dataset and model, we aim to position HTS classification as a new community benchmark task and invite future work in retrieval, reasoning, and alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18384v1">AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18344v1">Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SubSpec, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1x speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5x speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18314v1">Exploiting Tree Structure for Credit Assignment in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce \textbf{Prefix-to-Tree (P2T)}, a simple procedure that converts a group of responses into a prefix tree and computes \emph{nonparametric} prefix values \(V(s)\) by aggregating descendant outcomes. Built on P2T, we propose \textbf{TEMPO} (\emph{\textbf{T}ree-\textbf{E}stimated \textbf{M}ean Prefix Value for \textbf{P}olicy \textbf{O}ptimization}), a critic-free algorithm that augments the group-relative outcome signal of GRPO with \emph{branch-gated} temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with roughly the same wall-clock time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12543v2">Disambiguation in Conversational Question Answering in the Era of LLMs and Agents: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 14 pages, 2 figures, Accepted at EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, especially in agentic settings, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18085v1">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18083v1">Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      We introduce Reasoning Core, a new scalable environment for Reinforcement Learning with Verifiable Rewards (RLVR), designed to advance foundational symbolic reasoning in Large Language Models (LLMs). Unlike existing benchmarks that focus on games or isolated puzzles, Reasoning Core procedurally generates problems across core formal domains, including PDDL planning, first-order logic, context-free grammar parsing, causal reasoning, and system equation solving. The environment is built on key design principles of high-generality problem distributions, verification via external tools, and continuous difficulty control, which together provide a virtually infinite supply of novel training instances. Initial zero-shot evaluations with frontier LLMs confirm the difficulty of Reasoning Core's tasks, positioning it as a promising resource to improve the reasoning capabilities of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18063v1">ARK-V1: An LLM-Agent for Knowledge Graph Question Answering Requiring Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Work in Progess
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning abilities but rely on internalized knowledge that is often insufficient, outdated, or incorrect when trying to answer a question that requires specific domain knowledge. Knowledge Graphs (KGs) provide structured external knowledge, yet their complexity and multi-hop reasoning requirements make integration challenging. We present ARK-V1, a simple KG-agent that iteratively explores graphs to answer natural language queries. We evaluate several not fine-tuned state-of-the art LLMs as backbones for ARK-V1 on the CoLoTa dataset, which requires both KG-based and commonsense reasoning over long-tail entities. ARK-V1 achieves substantially higher conditional accuracies than Chain-of-Thought baselines, and larger backbone models show a clear trend toward better coverage, correctness, and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18058v1">Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18052v1">The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for social simulation, where populations of agents are expected to reproduce human-like collective behavior. However, we find that many recent studies adopt experimental designs that systematically undermine the validity of their claims. From a survey of over 40 papers, we identify six recurring methodological flaws: agents are often homogeneous (Profile), interactions are absent or artificially imposed (Interaction), memory is discarded (Memory), prompts tightly control outcomes (Minimal-Control), agents can infer the experimental hypothesis (Unawareness), and validation relies on simplified theoretical models rather than real-world data (Realism). For instance, GPT-4o and Qwen-3 correctly infer the underlying social experiment in 53.1% of cases when given instructions from prior work-violating the Unawareness principle. We formalize these six requirements as the PIMMUR principles and argue they are necessary conditions for credible LLM-based social simulation. To demonstrate their impact, we re-run five representative studies using a framework that enforces PIMMUR and find that the reported social phenomena frequently fail to emerge under more rigorous conditions. Our work establishes methodological standards for LLM-based multi-agent research and provides a foundation for more reliable and reproducible claims about "AI societies."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18015v1">Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, showing limited capacity to generalize to this novel task. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03116v2">Measuring Scalar Constructs in Social Science with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Many constructs that characterize language, like its complexity or emotionality, have a naturally continuous semantic structure; a public speech is not just "simple" or "complex," but exists on a continuum between extremes. Although large language models (LLMs) are an attractive tool for measuring scalar constructs, their idiosyncratic treatment of numerical outputs raises questions of how to best apply them. We address these questions with a comprehensive evaluation of LLM-based approaches to scalar construct measurement in social science. Using multiple datasets sourced from the political science literature, we evaluate four approaches: unweighted direct pointwise scoring, aggregation of pairwise comparisons, token-probability-weighted pointwise scoring, and finetuning. Our study finds that pairwise comparisons made by LLMs produce better measurements than simply prompting the LLM to directly output the scores, which suffers from bunching around arbitrary numbers. However, taking the weighted mean over the token probability of scores further improves the measurements over the two previous approaches. Finally, finetuning smaller models with as few as 1,000 training pairs can match or exceed the performance of prompted LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18761v2">How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 19 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      We introduce Grade School Math with Distracting Context (GSM-DC), a synthetic benchmark to evaluate Large Language Models' (LLMs) reasoning robustness against systematically controlled irrelevant context (IC). GSM-DC constructs symbolic reasoning graphs with precise distractor injections, enabling rigorous, reproducible evaluation. Our experiments demonstrate that LLMs are significantly sensitive to IC, affecting both reasoning path selection and arithmetic accuracy. Additionally, training models with strong distractors improves performance in both in-distribution and out-of-distribution scenarios. We further propose a stepwise tree search guided by a process reward model, which notably enhances robustness in out-of-distribution conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17998v1">Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted as Poster at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The efficiency of Bayesian optimization (BO) relies heavily on the choice of the Gaussian process (GP) kernel, which plays a central role in balancing exploration and exploitation under limited evaluation budgets. Traditional BO methods often rely on fixed or heuristic kernel selection strategies, which can result in slow convergence or suboptimal solutions when the chosen kernel is poorly suited to the underlying objective function. To address this limitation, we propose a freshly-baked Context-Aware Kernel Evolution (CAKE) to enhance BO with large language models (LLMs). Concretely, CAKE leverages LLMs as the crossover and mutation operators to adaptively generate and refine GP kernels based on the observed data throughout the optimization process. To maximize the power of CAKE, we further propose BIC-Acquisition Kernel Ranking (BAKER) to select the most effective kernel through balancing the model fit measured by the Bayesian information criterion (BIC) with the expected improvement at each iteration of BO. Extensive experiments demonstrate that our fresh CAKE-based BO method consistently outperforms established baselines across a range of real-world tasks, including hyperparameter optimization, controller tuning, and photonic chip design. Our code is publicly available at https://github.com/cake4bo/cake.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17946v1">HICode: Hierarchical Inductive Coding with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Long paper accepted at EMNLP 2025 main conference, 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Despite numerous applications for fine-grained corpus analysis, researchers continue to rely on manual labeling, which does not scale, or statistical tools like topic modeling, which are difficult to control. We propose that LLMs have the potential to scale the nuanced analyses that researchers typically conduct manually to large text corpora. To this effect, inspired by qualitative research methods, we develop HICode, a two-part pipeline that first inductively generates labels directly from analysis data and then hierarchically clusters them to surface emergent themes. We validate this approach across three diverse datasets by measuring alignment with human-constructed themes and demonstrating its robustness through automated and human evaluations. Finally, we conduct a case study of litigation documents related to the ongoing opioid crisis in the U.S., revealing aggressive marketing strategies employed by pharmaceutical companies and demonstrating HICode's potential for facilitating nuanced analyses in large-scale data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17932v1">Training-free Truthfulness Detection via Value Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models often generate factually incorrect outputs, motivating efforts to detect the truthfulness of their content. Most existing approaches rely on training probes over internal activations, but these methods suffer from scalability and generalization issues. A recent training-free method, NoVo, addresses this challenge by exploiting statistical patterns from the model itself. However, it focuses exclusively on attention mechanisms, potentially overlooking the MLP module-a core component of Transformer models known to support factual recall. In this paper, we show that certain value vectors within MLP modules exhibit truthfulness-related statistical patterns. Building on this insight, we propose TruthV, a simple and interpretable training-free method that detects content truthfulness by leveraging these value vectors. On the NoVo benchmark, TruthV significantly outperforms both NoVo and log-likelihood baselines, demonstrating that MLP modules-despite being neglected in prior training-free efforts-encode rich and useful signals for truthfulness detection. These findings offer new insights into how truthfulness is internally represented in LLMs and motivate further research on scalable and interpretable truthfulness detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01659v2">From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10369v2">SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Optimizing Register Transfer Level (RTL) code is crucial for improving the power, performance, and area (PPA) of digital circuits in the early stages of synthesis. Manual rewriting, guided by synthesis feedback, can yield high-quality results but is time-consuming and error-prone. Most existing compiler-based approaches have difficulty handling complex design constraints. Large Language Model (LLM)-based methods have emerged as a promising alternative to address these challenges. However, LLM-based approaches often face difficulties in ensuring alignment between the generated code and the provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL optimization framework that seamlessly integrates LLM-based code rewriting with symbolic reasoning techniques. Our method incorporates a retrieval-augmented generation (RAG) system of optimization rules and Abstract Syntax Tree (AST)-based templates, enabling LLM-based rewriting that maintains syntactic correctness while minimizing undesired circuit behaviors. A symbolic module is proposed for analyzing and optimizing finite state machine (FSM) logic, allowing fine-grained state merging and partial specification handling beyond the scope of pattern-based compilers. Furthermore, a fast verification pipeline, combining formal equivalence checks with test-driven validation, further reduces the complexity of verification. Experiments on the RTL-Rewriter benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%, respectively, compared to the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17901v1">Does Audio Matter for Modern Video-LLMs and Their Benchmarks?</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 5 pages, 2 figures, under review. Project page: https://github.com/naver-ai/LLaVA-AV-SSM
    </div>
    <details class="paper-abstract">
      Modern multimodal large language models often claim "video understanding," yet most evaluations use muted videos or simply discard audio. We ask a direct question: how much does audio actually matter for contemporary Video-LLMs and the benchmarks that certify them? We audit widely used suites and observe that many items are even solvable from a single frame, rendering audio largely redundant. Building on LLaVA-OneVision architecture, we attach a speech/audio encoder (e.g., Whisper) and analyze when audio helps, while addressing audio token explosion with a lightweight Mamba-based state-space token compressor. We find that audio yields minimal gains on recent video benchmarks but is decisive on curated, audio-sensitive subsets. To enable faithful evaluation, we release AVQA-Hard and Music-AVQA-Hard, our model, and code. Our findings surface a growing gap between current academic practice and real-world expectations, and provide practical tools for scalable audio-visual Video-LLMs. We will fully open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14498v4">LLaSA: A Sensor-Aware LLM for Natural Language Reasoning of Human Activity from IMU Data</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Wearable systems can recognize activities from IMU data but often fail to explain their underlying causes or contextual significance. To address this limitation, we introduce two large-scale resources: SensorCap, comprising 35,960 IMU--caption pairs, and OpenSQA, with 199,701 question--answer pairs designed for causal and explanatory reasoning. OpenSQA includes a curated tuning split (Tune-OpenSQA) optimized for scientific accuracy, narrative clarity, and diagnostic insight. Leveraging these datasets, we develop LLaSA (Large Language and Sensor Assistant), a family of compact sensor-aware language models (7B and 13B) that generate interpretable, context-rich responses to open-ended questions grounded in raw IMU data. LLaSA outperforms commercial LLMs, including GPT-3.5 and GPT-4o-mini, on benchmark and real-world tasks, demonstrating the effectiveness of domain supervision and model alignment for sensor reasoning. Our code repository and datasets can be found at https://github.com/BASHLab/LLaSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17796v1">Findings of the Fourth Shared Task on Multilingual Coreference Resolution: Can LLMs Dethrone Traditional Approaches?</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to CODI-CRAC 2025
    </div>
    <details class="paper-abstract">
      The paper presents an overview of the fourth edition of the Shared Task on Multilingual Coreference Resolution, organized as part of the CODI-CRAC 2025 workshop. As in the previous editions, participants were challenged to develop systems that identify mentions and cluster them according to identity coreference. A key innovation of this year's task was the introduction of a dedicated Large Language Model (LLM) track, featuring a simplified plaintext format designed to be more suitable for LLMs than the original CoNLL-U representation. The task also expanded its coverage with three new datasets in two additional languages, using version 1.3 of CorefUD - a harmonized multilingual collection of 22 datasets in 17 languages. In total, nine systems participated, including four LLM-based approaches (two fine-tuned and two using few-shot adaptation). While traditional systems still kept the lead, LLMs showed clear potential, suggesting they may soon challenge established approaches in future editions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11405v2">DCR: Quantifying Data Contamination in LLMs Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data during the training process, inflating performance metrics, and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC risk across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04931v2">Breaking the News: Taking the Roles of Influencer vs. Journalist in a LLM-Based Game for Raising Misinformation Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to ACM CHI PLAY 2025
    </div>
    <details class="paper-abstract">
      Effectively mitigating online misinformation requires understanding of their mechanisms and learning of practical skills for identification and counteraction. Serious games may serve as tools for combating misinformation, teaching players to recognize common misinformation tactics, and improving their skills of discernment. However, current interventions are designed as single-player, choice-based games, which present players with limited predefined choices. Such restrictions reduce replayability and may lead to an overly simplistic understanding of misinformation and how to debunk them. This study seeks to empower people to understand opinion-influencing and misinformation-debunking processes. We created a Player vs. Player (PvP) game in which participants attempt to generate or debunk misinformation to convince the public opinion represented by LLM. Using a within-subjects mixed-methods study design (N=47), we found that this game significantly raised participants' media literacy and improved their ability to identify misinformation. Qualitative analyses revealed how participants' use of debunking and content creation strategies deepened their understanding of misinformation. This work shows the potential for illuminating contrasting viewpoints of social issues by LLM-based mechanics in PvP games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21693v3">MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by EMNLP 2025 Findings, 33 pages, 30 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14746v4">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Trained LLMs in the transformer architecture are typically sparse in that most of the parameters are negligible, raising questions on efficiency. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Liebler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^{\gamma}$ where $D$ is the size of the training data and $ \gamma \in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17730v1">ConfClip: Confidence-Weighted and Clipped Reward for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a standard paradigm for refining large language models (LLMs) beyond pre-training and instruction tuning. A prominent line of work is RL with verifiable rewards (RLVR), which leverages automatically verifiable outcomes (e.g., correctness or executability) to generate reward signals. While efficient, this framework faces two key limitations: First, its binary feedback is too sparse to capture the quality of the reasoning process. Second, its coarse-grained rewards potentially lead to vanishing gradients. Inspired by observations from human learning, we introduce a RL technique that integrates verifiable outcomes with the model's own confidence estimates. This joint design enriches the reward signal, providing finer-grained feedback and implicitly supervising the reasoning process. Experimental results demonstrate that our proposed method enhances RL performance across multiple datasets and reduces token consumption during inference, while incurring negligible additional training cost. Moreover, it can be used as a plug-in module to enhance other state-of-the-art RL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12260v4">LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22424v2">CoSIL: Issue Localization via LLM-Driven Code Graph Searching</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      Issue solving aims to generate patches to fix reported issues in real-world code repositories according to issue descriptions. Issue localization forms the basis for accurate issue solving. Recently, LLM-based issue localization methods have demonstrated state-of-the-art performance. However, these methods either search from files mentioned in issue descriptions or in the whole repository and struggle to balance the breadth and depth of the search space to converge on the target efficiently. Moreover, they allow LLM to explore whole repositories freely, making it challenging to control the search direction to prevent the LLM from searching for incorrect targets. This paper introduces CoSIL, an LLM-driven, powerful function-level issue localization method without training or indexing. CoSIL employs a two-phase code graph search strategy. It first conducts broad exploration at the file level using dynamically constructed module call graphs, and then performs in-depth analysis at the function level by expanding the module call graph into a function call graph and executing iterative searches. To precisely control the search direction, CoSIL designs a pruner to filter unrelated directions and irrelevant contexts. To avoid incorrect interaction formats in long contexts, CoSIL introduces a reflection mechanism that uses additional independent queries in short contexts to enhance formatted abilities. Experiment results demonstrate that CoSIL achieves a Top-1 localization accuracy of 43.3\% and 44.6\% on SWE-bench Lite and SWE-bench Verified, respectively, with Qwen2.5-Coder-32B, average outperforming the state-of-the-art methods by 96.04\%. When CoSIL is integrated into an issue-solving method, Agentless, the issue resolution rate improves by 2.98\%--30.5\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17703v1">An LLM-based Agent Simulation Approach to Study Moral Evolution</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      The evolution of morality presents a puzzle: natural selection should favor self-interest, yet humans developed moral systems promoting altruism. We address this question by introducing a novel Large Language Model (LLM)-based agent simulation framework modeling prehistoric hunter-gatherer societies. This platform is designed to probe diverse questions in social evolution, from survival advantages to inter-group dynamics. To investigate moral evolution, we designed agents with varying moral dispositions based on the Expanding Circle Theory \citep{singer1981expanding}. We evaluated their evolutionary success across a series of simulations and analyzed their decision-making in specially designed moral dilemmas. These experiments reveal how an agent's moral framework, in combination with its cognitive constraints, directly shapes its behavior and determines its evolutionary outcome. Crucially, the emergent patterns echo seminal theories from related domains of social science, providing external validation for the simulations. This work establishes LLM-based simulation as a powerful new paradigm to complement traditional research in evolutionary biology and anthropology, opening new avenues for investigating the complexities of moral and social evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17701v1">Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at edu4AI'25: 2nd Workshop on Education for Artificial Intelligence | co-located with ECAI, October 26th, 2025, Bologna, Italy. 7 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17694v1">Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted for publication at the 18th International Natural Language Generation Conference (INLG 2025)
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09996v3">From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NeurIPS 2025 Accepted Paper
    </div>
    <details class="paper-abstract">
      Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17628v1">MSCoRe: A Benchmark for Multi-Stage Collaborative Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in question-answering (QA) tasks within single domains. However, their reasoning and coordination capabilities in complex, multi-stage scenarios remain underexplored. Existing benchmarks typically focus on isolated tasks or narrow domains, overlooking models' abilities for multi-stage collaboration and optimization without explicit external guidance. To bridge this gap, we propose \textbf{MSCoRe}, a novel benchmark comprising 126696 domain-specific QA instances spanning scenarios in automotive, pharmaceutical, electronics, and energy sectors. The dataset is created using a structured three-phase pipeline: dynamic sampling, iterative question-answer generation, and a multi-level quality assessment to ensure data quality. Tasks are further categorized into three difficulty levels according to stage coverage and complexity. With MSCoRe, we have conducted a comprehensive evaluation of various state-of-the-art LLM agents. The commercial models performed best across all tasks and scenarios, but a notable gap in ROUGE scores remains between simple and complex tasks. We also tested the models' robustness and found that their performance is negatively affected by noisy data. MSCoRe provides a valuable new resource for the community to evaluate and improve multi-stage reasoning in LLM agents. The code and data are available at https://github.com/D3E0-source/MSCoRE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17552v1">Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NIPS 2025
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14423v2">Scaling Low-Resource MT via Synthetic Data Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We investigate the potential of LLM-generated synthetic data for improving low-resource Machine Translation (MT). Focusing on seven diverse target languages, we construct a document-level synthetic corpus from English Europarl, and extend it via pivoting to 147 additional language pairs. Automatic and human evaluation confirm its overall high quality. We study its practical application by (i) identifying effective training regimes, (ii) comparing our data with the HPLT dataset, (iii) studying the effect of varying training data size, and (iiii) testing its utility beyond English-centric MT. Finally, we introduce SynOPUS, a public repository for synthetic parallel datasets. Our findings show that LLM-generated synthetic data, even when noisy, can substantially improve MT performance for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17505v1">CorefInst: Leveraging LLMs for Multilingual Coreference Resolution</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted for publication in Transactions of the Association for Computational Linguistics (TACL) (2025 August). Submission: March, 2025. Revision: July, 2025. Acceptance: August, 2025
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a crucial yet challenging task in natural language understanding, often constrained by task-specific architectures and encoder-based language models that demand extensive training and lack adaptability. This study introduces the first multilingual CR methodology which leverages decoder-only LLMs to handle both overt and zero mentions. The article explores how to model the CR task for LLMs via five different instruction sets using a controlled inference method. The approach is evaluated across three LLMs; Llama 3.1, Gemma 2, and Mistral 0.3. The results indicate that LLMs, when instruction-tuned with a suitable instruction set, can surpass state-of-the-art task-specific architectures. Specifically, our best model, a fully fine-tuned Llama 3.1 for multilingual CR, outperforms the leading multilingual CR model (i.e., Corpipe 24 single stage variant) by 2 pp on average across all languages in the CorefUD v1.2 dataset collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v2">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14615v2">SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a puzzle using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-based and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. Our error analysis reveals systematic failures such as satisfiability bias, context inconsistency, and condition omission, highlighting limitations of current LLMs in search-based logical reasoning. Our code and data are publicly available at https://github.com/Anjiang-Wei/SATBench
    </details>
</div>
