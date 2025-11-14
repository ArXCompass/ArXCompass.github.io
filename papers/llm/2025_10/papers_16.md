# llm - 2025_10

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- Part 16
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)
- [Part 20](papers_20.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06092v1">Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06078v1">Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06039v1">CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06018v1">Evaluating The Impact of Stimulus Quality in Investigations of LLM Language Performance</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at https://brigap-workshop.github.io/ Information to be updated upon publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies employing Large Language Models (LLMs) to test the Argument from the Poverty of the Stimulus (APS) have yielded contrasting results across syntactic phenomena. This paper investigates the hypothesis that characteristics of the stimuli used in recent studies, including lexical ambiguities and structural complexities, may confound model performance. A methodology is proposed for re-evaluating LLM competence on syntactic prediction, focusing on GPT-2. This involves: 1) establishing a baseline on previously used (both filtered and unfiltered) stimuli, and 2) generating a new, refined dataset using a state-of-the-art (SOTA) generative LLM (Gemini 2.5 Pro Preview) guided by linguistically-informed templates designed to mitigate identified confounds. Our preliminary findings indicate that GPT-2 demonstrates notably improved performance on these refined PG stimuli compared to baselines, suggesting that stimulus quality significantly influences outcomes in surprisal-based evaluations of LLM syntactic competency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02418v2">BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about pop-up banner closure. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06001v1">Exploring Gaps in the APS: Direct Minimal Pair Analysis in LLM Syntactic Assessments</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at the https://brigap-workshop.github.io/ Information to be updated after publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies probing the Argument from the Poverty of the Stimulus (APS) have applied Large Language Models (LLMs) to test the learnability of complex syntax through surprisal-based metrics. However, divergent conclusions raise questions concerning the insights these metrics offer. While Wilcox et al. (2024) used direct minimal pair comparisons (the "wh-effect") to demonstrate that models successfully generalise knowledge of filler-gap dependencies, Lan et al. (2024) used a Difference-in-Differences (DiD) metric and found that models largely fail on parasitic gaps (PGs). This paper argues that the direct minimal pair approach offers greater diagnostic transparency. We demonstrate this by generating a full 8-permutation paradigm of refined PG stimuli and evaluating the GPT-2 model used in previous studies with a systematic Wilcox-style wh-effect analysis. Our results show that GPT-2 succeeds across all four tested conditions, indicating robust knowledge of filler-gap licensing principles even in complex PG environments. This finding, which contrasts with the more ambiguous results from DiD-style metrics, suggests that the choice of evaluation metric is critical for assessing an LLM's syntactic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05987v1">Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to complex tasks that require extended reasoning. In such settings, models often benefit from diverse chains-of-thought to arrive at multiple candidate solutions. This requires two competing objectives: to inject enough stochasticity to explore multiple reasoning chains, and to ensure sufficient accuracy and quality in each path. Existing works pursue the first objective by increasing exploration at highly uncertain steps with higher temperature or larger candidate token sets, while others improve reliability by rejecting samples with low confidence post-generation, implying that low confidence correlates with low answer quality. These two lines of thought are in conflict, as they conflate different sources of uncertainty. To resolve this, we argue that the decoding rule should be calibrated by correctness, not confidence alone. We should sample from tokens with higher estimated correctness, and reduce sampling where expected correctness is low. We propose simple strategies that achieve this goal: Greedy-Threshold makes sampling greedy at very low confidence steps. Calibrated-TopK and Calibrated-epsilon set truncation threshold based on estimated rank-wise correctness. Together, our findings challenge prevailing heuristics about decoding under uncertainty and show gains across math and general reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05950v1">Training-Free Time Series Classification via In-Context Reasoning with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 8 pages main content, 12 pages total including appendix, 1 figure
    </div>
    <details class="paper-abstract">
      Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at https://github.com/SongyuanSui/FETATSC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v3">Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Published in IEEE Access
    </div>
    <details class="paper-abstract">
      Recent studies have investigated whether large language models (LLMs) can support obscured communication, which is characterized by core aspects such as inferring subtext and evading suspicions. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two limitations with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these limitations, we propose a microscopic and systematic approach to the investigation. Specifically, we introduce six fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05935v1">LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05909v1">Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Open-source code available at https://github.com/flowersteam/llm_persuasion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v2">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v3">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05869v1">The fragility of "cultural tendencies" in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      In a recent study, Lu, Song, and Zhang (2025) (LSZ) propose that large language models (LLMs), when prompted in different languages, display culturally specific tendencies. They report that the two models (i.e., GPT and ERNIE) respond in more interdependent and holistic ways when prompted in Chinese, and more independent and analytic ways when prompted in English. LSZ attribute these differences to deep-seated cultural patterns in the models, claiming that prompt language alone can induce substantial cultural shifts. While we acknowledge the empirical patterns they observed, we find their experiments, methods, and interpretations problematic. In this paper, we critically re-evaluate the methodology, theoretical framing, and conclusions of LSZ. We argue that the reported "cultural tendencies" are not stable traits but fragile artifacts of specific models and task design. To test this, we conducted targeted replications using a broader set of LLMs and a larger number of test items. Our results show that prompt language has minimal effect on outputs, challenging LSZ's claim that these models encode grounded cultural beliefs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05864v1">Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19056v2">An Embarrassingly Simple Defense Against LLM Abliteration Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 preprint - under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14146v3">MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05758v1">EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Under review for ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2510.03930v1">LLM Chemistry Estimation for Multi-LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 20 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2308.03688v3">AgentBench: Evaluating LLMs as Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Published in ICLR 2024
    </div>
    <details class="paper-abstract">
      The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities. Our extensive test over \num API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and many OSS competitors that are no larger than 70B. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Improving instruction following and training on high quality multi-round alignment data could improve agent performance. And different from existing assumptions, training on code present ambivalent impacts on different agent tasks. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00927v3">Text Clustering as Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Text clustering serves as a fundamental technique for organizing and interpreting unstructured textual data, particularly in contexts where manual annotation is prohibitively costly. With the rapid advancement of Large Language Models (LLMs) and their demonstrated effectiveness across a broad spectrum of NLP tasks, an emerging body of research has begun to explore their potential in the domain of text clustering. However, existing LLM-based approaches still rely on fine-tuned embedding models and sophisticated similarity metrics, rendering them computationally intensive and necessitating domain-specific adaptation. To address these limitations, we propose a novel framework that reframes text clustering as a classification task by harnessing the in-context learning capabilities of LLMs. Our framework eliminates the need for fine-tuning embedding models or intricate clustering algorithms. It comprises two key steps: first, the LLM is prompted to generate a set of candidate labels based on the dataset and then merges semantically similar labels; second, it assigns the most appropriate label to each text sample. By leveraging the advanced natural language understanding and generalization capabilities of LLMs, the proposed approach enables effective clustering with minimal human intervention. Experimental results on diverse datasets demonstrate that our framework achieves comparable or superior performance to state-of-the-art embedding-based clustering techniques, while significantly reducing computational complexity and resource requirements. These findings underscore the transformative potential of LLMs in simplifying and enhancing text clustering tasks. We make our code available to the public for utilization at https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM. We also provide the supplementary Appendix within the repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05748v1">Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05742v1">Vipera: Blending Visual and LLM-Driven Guidance for Systematic Auditing of Text-to-Image Generative AI</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Despite their increasing capabilities, text-to-image generative AI systems are known to produce biased, offensive, and otherwise problematic outputs. While recent advancements have supported testing and auditing of generative AI, existing auditing methods still face challenges in supporting effectively explore the vast space of AI-generated outputs in a structured way. To address this gap, we conducted formative studies with five AI auditors and synthesized five design goals for supporting systematic AI audits. Based on these insights, we developed Vipera, an interactive auditing interface that employs multiple visual cues including a scene graph to facilitate image sensemaking and inspire auditors to explore and hierarchically organize the auditing criteria. Additionally, Vipera leverages LLM-powered suggestions to facilitate exploration of unexplored auditing directions. Through a controlled experiment with 24 participants experienced in AI auditing, we demonstrate Vipera's effectiveness in helping auditors navigate large AI output spaces and organize their analyses while engaging with diverse criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05733v1">Syn-Diag: An LLM-based Synergistic Framework for Generalizable Few-shot Fault Diagnosis on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Industrial fault diagnosis faces the dual challenges of data scarcity and the difficulty of deploying large AI models in resource-constrained environments. This paper introduces Syn-Diag, a novel cloud-edge synergistic framework that leverages Large Language Models to overcome these limitations in few-shot fault diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic Synergy, which aligns signal features with the LLM's semantic space through cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically constructs contextual prompts to enhance diagnostic accuracy with limited samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create a lightweight, efficient edge model capable of online updates via a shared decision space. Extensive experiments on six datasets covering different CWRU and SEU working conditions show that Syn-Diag significantly outperforms existing methods, especially in 1-shot and cross-condition scenarios. The edge model achieves performance comparable to the cloud version while reducing model size by 83% and latency by 50%, offering a practical, robust, and deployable paradigm for modern intelligent diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04093v2">Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14913v3">Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05709v1">Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05703v1">Primal-Dual Direct Preference Optimization for Constrained LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The widespread application of Large Language Models (LLMs) imposes increasing demands on safety, such as reducing harmful content and fake information, and avoiding certain forbidden tokens due to rules and laws. While there have been several recent works studying safe alignment of LLMs, these works either require the training of reward and cost models and incur high memory and computational costs, or need prior knowledge about the optimal solution. Motivated by this fact, we study the problem of constrained alignment in LLMs, i.e., maximizing the output reward while restricting the cost due to potentially unsafe content to stay below a threshold. For this problem, we propose a novel primal-dual DPO approach, which first trains a model using standard DPO on reward preference data to provide reward information, and then adopts a rearranged Lagrangian DPO objective utilizing the provided reward information to fine-tune LLMs on cost preference data. Our approach significantly reduces memory and computational costs, and does not require extra prior knowledge. Moreover, we establish rigorous theoretical guarantees on the suboptimality and constraint violation of the output policy. We also extend our approach to an online data setting by incorporating exploration bonuses, which enables our approach to explore uncovered prompt-response space, and then provide theoretical results that get rid of the dependence on preference data coverage. Experimental results on the widely-used preference dataset PKU-SafeRLHF demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05632v1">From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), the demand for high-performance LLM inference services continues to grow. To meet this demand, a growing number of AI accelerators have been proposed, such as Google TPU, Huawei NPU, Graphcore IPU, and Cerebras WSE, etc. Most of these accelerators adopt multi-core architectures to achieve enhanced scalability, but lack the flexibility of SIMT architectures. Therefore, without careful configuration of the hardware architecture, as well as deliberate design of tensor parallelism and core placement strategies, computational resources may be underutilized, resulting in suboptimal inference performance. To address these challenges, we first present a multi-level simulation framework with both transaction-level and performance-model-based simulation for multi-core NPUs. Using this simulator, we conduct a systematic analysis and further propose the optimal solutions for tensor parallelism strategies, core placement policies, memory management methods, as well as the selection between PD-disaggregation and PD-fusion on multi-core NPUs. We conduct comprehensive experiments on representative LLMs and various NPU configurations. The evaluation results demonstrate that, our solution can achieve 1.32x-6.03x speedup compared to SOTA designs for multi-core NPUs across different hardware configurations. As for LLM serving, our work offers guidance on designing optimal hardware architectures and serving strategies for multi-core NPUs across various LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18851v2">Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Identifying LLM-generated code through watermarking poses a challenge in preserving functional correctness. Previous methods rely on the assumption that watermarking high-entropy tokens effectively maintains output quality. Our analysis reveals a fundamental limitation of this assumption: syntax-critical tokens such as keywords often exhibit the highest entropy, making existing approaches vulnerable to logic corruption. We present STONE, a syntax-aware watermarking method that embeds watermarks only in non-syntactic tokens and preserves code integrity. For its rigorous assessment, we also introduce STEM, a comprehensive framework that balances three critical dimensions: correctness, detectability, and imperceptibility. Across Python, C++, and Java, STONE preserves correctness, sustains strong detectability, and achieves balanced performance with minimal overhead. Our implementation is available at https://anonymous.4open.science/r/STONE-watermarking-AB4B/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05605v1">AutoPentester: An LLM Agent-based Framework for Automated Pentesting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 IEEE TrustCom 2025 10 pages
    </div>
    <details class="paper-abstract">
      Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05598v1">AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11182v3">A Middle Path for On-Premises LLM Deployment: Preserving Privacy Without Sacrificing Model Confidentiality</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 8 pages for main content of the paper
    </div>
    <details class="paper-abstract">
      Privacy-sensitive users require deploying large language models (LLMs) within their own infrastructure (on-premises) to safeguard private data and enable customization. However, vulnerabilities in local environments can lead to unauthorized access and potential model theft. To address this, prior research on small models has explored securing only the output layer within hardware-secured devices to balance model confidentiality and customization. Yet this approach fails to protect LLMs effectively. In this paper, we discover that (1) query-based distillation attacks targeting the secured top layer can produce a functionally equivalent replica of the victim model; (2) securing the same number of layers, bottom layers before a transition layer provide stronger protection against distillation attacks than top layers, with comparable effects on customization performance; and (3) the number of secured layers creates a trade-off between protection and customization flexibility. Based on these insights, we propose SOLID, a novel deployment framework that secures a few bottom layers in a secure environment and introduces an efficient metric to optimize the trade-off by determining the ideal number of hidden layers. Extensive experiments on five models (1.3B to 70B parameters) demonstrate that SOLID outperforms baselines, achieving a better balance between protection and downstream customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00965v2">FLEx: Personalized Federated Learning for Mixture-of-Experts LLMs via Expert Grafting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Federated instruction tuning of large language models (LLMs) is challenged by significant data heterogeneity across clients, demanding robust personalization. The Mixture of Experts (MoE) architecture, where experts can specialize in distinct data patterns, presents a natural architectural solution to this challenge. The inherent sparsity of the MoE architecture, achieved by selectively activating experts, poses a significant challenge to its integration with federated learning (FL). Conventional FL frameworks, designed for dense models, naively aggregate all expert parameters irrespective of their local activation patterns. This naive approach not only undermines MoE's dynamic sparsity but also risks corrupting the world knowledge within pretrained experts. To address this, we propose FLEx (Federated LLMs with Personalized Experts), a novel framework that leverages pretrained MoE-based LLMs for efficient personalization. By aggregating only the shared non-expert parameters, FLEx significantly reduces communication overhead and preserves the world knowledge stored within the frozen pretrained experts. For personalization, we introduce a novel expert grafting mechanism that leverages dynamic sparsity to construct a client-specific expert from selected components of pretrained experts, tailored to local data. This grafted expert is then fine-tuned locally alongside the gating mechanism. This joint training enables the model to learn when to leverage the shared knowledge from frozen experts and when to employ the personalized one. Evaluations on diverse, non-IID instruction tuning datasets show that FLEx consistently outperforms federated baselines on average, while demonstrating strong knowledge preservation on the knowledge-driven benchmark MMLU. Our code is available at \href{https://anonymous.4open.science/r/FLEx-8F12}{\texttt{https://anonymous.4open.science/r/FLEx-8F12}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05582v1">(Token-Level) \textbf{InfoRMIA}: Stronger Membership Inference and Memorization Assessment for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency. In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05577v1">Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Recent advancements in language agents have led to significant improvements in multi-hop reasoning tasks. However, existing approaches often struggle with handling open-domain problems, which require massive information retrieval due to their reliance on a fixed sequence of actions. To address this, we propose Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive strategies for information exploration in open-domain multi-hop reasoning tasks. Our approach begins by identifying key entities relevant to the problem, which serve as the initial nodes in the reasoning process. From these initial nodes, we then generate reasoning child nodes with the process being refined through a combination of historical error analysis and real-time feedback, which allows the framework to dynamically adjust and optimize its reasoning strategies. By integrating depth-first search with an innovative node generation technique, our framework adapts based on both prior error paths and concurrently generated nodes at the same hierarchical level. This dynamic strategy effectively expands the search space while ensuring the reasoning process systematically converges toward accurate solutions. Experimental results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and 7.25% respectively, highlighting its versatility and potential to enhance language agents in multi-hop reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08833v2">Position: The Pitfalls of Over-Alignment: Overly Caution Health-Related Responses From LLMs are Unethical and Dangerous</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are usually aligned with "human values/preferences" to prevent harmful output. Discussions around the alignment of Large Language Models (LLMs) generally focus on preventing harmful outputs. However, in this paper, we argue that in health-related queries, over-alignment-leading to overly cautious responses-can itself be harmful, especially for people with anxiety and obsessive-compulsive disorder (OCD). This is not only unethical but also dangerous to the user, both mentally and physically. We also showed qualitative results that some LLMs exhibit varying degrees of alignment. Finally, we call for the development of LLMs with stronger reasoning capabilities that provide more tailored and nuanced responses to health queries. Warning: This paper contains materials that could trigger health anxiety or OCD. Dataset and full results can be found in https://github.com/weathon/over-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25795v2">Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 7 pages, 1 figure. This paper has been accepted to the IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2025), Georgia Institute of Technology, Atlanta, Georgia, October 26-29, 2025
    </div>
    <details class="paper-abstract">
      This paper investigates algorithmic bias in language-based models for automated depression detection, focusing on socio-demographic disparities related to gender and race/ethnicity. Models trained using deep neural networks (DNN) based embeddings are compared to few-shot learning approaches with large language models (LLMs), evaluating both performance and fairness on clinical interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz (DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to DNN-based models, while in-context learning with varied prompt framing and shot counts is explored for LLMs. Results indicate that LLMs outperform DNN-based models in depression classification, particularly for underrepresented groups such as Hispanic participants. LLMs also exhibit reduced gender bias compared to DNN-based embeddings, though racial disparities persist. Among fairness-aware techniques for mitigating bias in DNN-based embeddings, the worst-group loss, which is designed to minimize loss for the worst-performing demographic group, achieves a better balance between performance and fairness. In contrast, the fairness-regularized loss minimizes loss across all groups but performs less effectively. In LLMs, guided prompting with ethical framing helps mitigate gender bias in the 1-shot setting. However, increasing the number of shots does not lead to further reductions in disparities. For race/ethnicity, neither prompting strategy nor increasing $N$ in $N$-shot learning effectively reduces disparities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05544v1">Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23058v3">Risk Profiling and Modulation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05520v1">CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13360v2">What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Prompt underspecification is a common challenge when interacting with LLMs. In this paper, we present an in-depth analysis of this problem, showing that while LLMs can often infer unspecified requirements by default (41.1%), such behavior is fragile: Under-specified prompts are 2x as likely to regress across model or prompt changes, sometimes with accuracy drops exceeding 20%. This instability makes it difficult to reliably build LLM applications. Moreover, simply specifying all requirements does not consistently help, as models have limited instruction-following ability and requirements can conflict. Standard prompt optimizers likewise provide little benefit. To address these issues, we propose requirements-aware prompt optimization mechanisms that improve performance by 4.8% on average over baselines. We further advocate for a systematic process of proactive requirements discovery, evaluation, and monitoring to better manage prompt underspecification in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05497v1">Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B- 671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3X and 4.0X average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {https://huggingface.co/datasets/core12345/MoE_expert_selection_trace. We will also release our simulation framework shortly to facilitate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05484v1">Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming part of tools used by children; however, existing benchmarks fail to capture how these models manage language, reasoning, and safety needs that are specific to various ages. We present ChildSafe, a benchmark that evaluates LLM safety through simulated child agents that embody four developmental stages. These agents, grounded in developmental psychology, enable a systematic study of child safety without the ethical implications of involving real children. ChildSafe assesses responses across nine safety dimensions (including privacy, misinformation, and emotional support) using age-weighted scoring in both sensitive and neutral contexts. Multi-turn experiments with multiple LLMs uncover consistent vulnerabilities that vary by simulated age, exposing shortcomings in existing alignment practices. By releasing agent templates, evaluation protocols, and an experimental corpus, we provide a reproducible framework for age-aware safety research. We encourage the community to expand this work with real child-centered data and studies, advancing the development of LLMs that are genuinely safe and developmentally aligned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05480v1">Vul-R2: A Reasoning LLM for Automated Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 13 pages, 8 figures. This paper is accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18356v2">The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 MRL Workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07218v2">LLM Unlearning via Neural Activation Redirection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The ability to selectively remove knowledge from LLMs is highly desirable. However, existing methods often struggle with balancing unlearning efficacy and retain model utility, and lack controllability at inference time to emulate base model behavior as if it had never seen the unlearned data. In this paper, we propose LUNAR, a novel unlearning method grounded in the Linear Representation Hypothesis and operates by redirecting the representations of unlearned data to activation regions that expresses its inability to answer. We show that contrastive features are not a prerequisite for effective activation redirection, and LUNAR achieves state-of-the-art unlearning performance and superior controllability. Specifically, LUNAR achieves between 2.9x and 11.7x improvement in the combined unlearning efficacy and model utility score (Deviation Score) across various base models and generates coherent, contextually appropriate responses post-unlearning. Moreover, LUNAR effectively reduces parameter updates to a single down-projection matrix, a novel design that significantly enhances efficiency by 20x and robustness. Finally, we demonstrate that LUNAR is robust to white-box adversarial attacks and versatile in real-world scenarios, including handling sequential unlearning requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01213v2">Show or Tell? Modeling the evolution of request-making in Human-LLM conversations</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Designing user-centered LLM systems requires understanding how people use them, but patterns of user behavior are often masked by the variability of queries. In this work, we introduce a new framework to describe request-making that segments user input into request content, roles assigned, query-specific context, and the remaining task-independent expressions. We apply the workflow to create and analyze a dataset of 211k real-world queries based on WildChat. Compared with similar human-human setups, we find significant differences in the language for request-making in the human-LLM scenario. Further, we introduce a novel and essential perspective of diachronic analyses with user expressions, which reveals fundamental and habitual user-LLM interaction patterns beyond individual task completion. We find that query patterns evolve from early ones emphasizing sole requests to combining more context later on, and individual users explore expression patterns but tend to converge with more experience. From there, we propose to understand communal trends of expressions underlying distinct tasks and discuss the preliminary findings. Finally, we discuss the key implications for user studies, computational pragmatics, and LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06477v1">Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Attention sinks and compression valleys have attracted significant attention as two puzzling phenomena in large language models, but have been studied in isolation. In this work, we present a surprising connection between attention sinks and compression valleys, tracing both to the formation of massive activations in the residual stream. We prove theoretically that massive activations necessarily produce representational compression and establish bounds on the resulting entropy reduction. Through experiments across several models (410M-120B parameters), we confirm that when the beginning-of-sequence token develops extreme activation norms in the middle layers, both compression valleys and attention sinks emerge simultaneously. Targeted ablation studies validate our theoretical predictions. This unified view motivates us to propose the Mix-Compress-Refine theory of information flow, as an attempt to explain how LLMs organize their computation in depth by controlling attention and representational compression via massive activations. Specifically, we posit that Transformer-based LLMs process tokens in three distinct phases: (1) broad mixing in the early layers, (2) compressed computation with limited mixing in the middle layers, and (3) selective refinement in the late layers. Our framework helps explain why embedding tasks perform best at intermediate layers, whereas generation tasks benefit from full-depth processing, clarifying differences in task-dependent representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23058v3">Risk Profiling and Modulation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.00927v3">Text Clustering as Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Text clustering serves as a fundamental technique for organizing and interpreting unstructured textual data, particularly in contexts where manual annotation is prohibitively costly. With the rapid advancement of Large Language Models (LLMs) and their demonstrated effectiveness across a broad spectrum of NLP tasks, an emerging body of research has begun to explore their potential in the domain of text clustering. However, existing LLM-based approaches still rely on fine-tuned embedding models and sophisticated similarity metrics, rendering them computationally intensive and necessitating domain-specific adaptation. To address these limitations, we propose a novel framework that reframes text clustering as a classification task by harnessing the in-context learning capabilities of LLMs. Our framework eliminates the need for fine-tuning embedding models or intricate clustering algorithms. It comprises two key steps: first, the LLM is prompted to generate a set of candidate labels based on the dataset and then merges semantically similar labels; second, it assigns the most appropriate label to each text sample. By leveraging the advanced natural language understanding and generalization capabilities of LLMs, the proposed approach enables effective clustering with minimal human intervention. Experimental results on diverse datasets demonstrate that our framework achieves comparable or superior performance to state-of-the-art embedding-based clustering techniques, while significantly reducing computational complexity and resource requirements. These findings underscore the transformative potential of LLMs in simplifying and enhancing text clustering tasks. We make our code available to the public for utilization at https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM. We also provide the supplementary Appendix within the repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05188v1">Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05186v1">OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Use Mathematical Programming to model Pipeline Parallelism with Offloading to balance efficiency and memory requirement
    </div>
    <details class="paper-abstract">
      Pipeline parallelism (PP) has become a standard technique for scaling large language model (LLM) training across multiple devices. However, despite recent progress in reducing memory consumption through activation offloading, existing approaches remain largely heuristic and coarse-grained, often overlooking the fine-grained trade-offs between memory, computation, and scheduling latency. In this work, we revisit the pipeline scheduling problem from a principled optimization perspective. We observe that prevailing strategies either rely on static rules or aggressively offload activations without fully leveraging the interaction between memory constraints and scheduling efficiency. To address this, we formulate scheduling as a constrained optimization problem that jointly accounts for memory capacity, activation reuse, and pipeline bubble minimization. Solving this model yields fine-grained schedules that reduce pipeline bubbles while adhering to strict memory budgets. Our approach complements existing offloading techniques: whereas prior approaches trade memory for time in a fixed pattern, we dynamically optimize the tradeoff with respect to model structure and hardware configuration. Experimental results demonstrate that our method consistently improves both throughput and memory utilization. In particular, we reduce idle pipeline time by up to 50% under the same per-device memory limit, and in some cases, enable the training of larger models within limited memory budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v3">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce answers with a single chain-of-thought, which restricts their ability to explore reasoning paths or self-correct flawed outputs in complex tasks. In this paper, we introduce MALT (Multi-Agent LLM Training), a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents. During data generation, each agent is repeatedly sampled to form a multi-agent search tree, where final outputs are graded against ground-truth data. We then apply value iteration to propagate reward signals back to each role-conditioned model, automatically producing multi-agent post-training data without human or teacher-model supervision. Our off-policy approach allows each agent to specialize by learning from correct and incorrect trajectories, ultimately improving the end-to-end reasoning chain. On MATH, GSM8K, and CSQA, MALT surpasses the same baseline LLM with a relative improvement of 15.66%, 7.42%, and 9.40% respectively, making it an important advance towards multi-agent cooperative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05087v1">TeachLM: Post-Training LLMs for Education Using Authentic Learning Data</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 28 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05069v1">SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Code: https://github.com/sdc17/SwiReasoning, Website: https://swireasoning.github.io/
    </div>
    <details class="paper-abstract">
      Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05052v1">Proactive defense against LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12491v3">Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 85% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05040v1">Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are trained flexibly to model extreme dependence in the data distribution; however, how to best utilize this information at inference time remains an open problem. In this work, we uncover an interesting property of these models: dLLMs trained on textual data implicitly learn a mixture of semi-autoregressive experts, where different generation orders reveal different specialized behaviors. We show that committing to any single, fixed inference time schedule, a common practice, collapses performance by failing to leverage this latent ensemble. To address this, we introduce HEX (Hidden semiautoregressive EXperts for test-time scaling), a training-free inference method that ensembles across heterogeneous block schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such as GSM8K, it boosts accuracy by up to 3.56X (from 24.72% to 88.10%), outperforming top-K margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX even yields significant gains on MATH benchmark from 16.40% to 40.00%, scientific reasoning on ARC-C from 54.18% to 87.80%, and TruthfulQA from 28.36% to 57.46%. Our results establish a new paradigm for test-time scaling in diffusion-based LLMs (dLLMs), revealing that the sequence in which masking is performed plays a critical role in determining performance during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08825v2">Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models are rapidly transforming social science research by enabling the automation of labor-intensive tasks like data annotation and text analysis. However, LLM outputs vary significantly depending on the implementation choices made by researchers (e.g., model selection or prompting strategy). Such variation can introduce systematic biases and random errors, which propagate to downstream analyses and cause Type I (false positive), Type II (false negative), Type S (wrong sign), or Type M (exaggerated effect) errors. We call this phenomenon where configuration choices lead to incorrect conclusions LLM hacking. We find that intentional LLM hacking is strikingly simple. By replicating 37 data annotation tasks from 21 published social science studies, we show that, with just a handful of prompt paraphrases, virtually anything can be presented as statistically significant. Beyond intentional manipulation, our analysis of 13 million labels from 18 different LLMs across 2361 realistic hypotheses shows that there is also a high risk of accidental LLM hacking, even when following standard research practices. We find incorrect conclusions in approximately 31% of hypotheses for state-of-the-art LLMs, and in half the hypotheses for smaller language models. While higher task performance and stronger general model capabilities reduce LLM hacking risk, even highly accurate models remain susceptible. The risk of LLM hacking decreases as effect sizes increase, indicating the need for more rigorous verification of LLM-based findings near significance thresholds. We analyze 21 mitigation techniques and find that human annotations provide crucial protection against false positives. Common regression estimator correction techniques can restore valid inference but trade off Type I vs. Type II errors. We publish a list of practical recommendations to prevent LLM hacking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04997v1">AutoEmpirical: LLM-Based Automated Research for Empirical Software Fault Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Understanding software faults is essential for empirical research in software development and maintenance. However, traditional fault analysis, while valuable, typically involves multiple expert-driven steps such as collecting potential faults, filtering, and manual investigation. These processes are both labor-intensive and time-consuming, creating bottlenecks that hinder large-scale fault studies in complex yet critical software systems and slow the pace of iterative empirical research. In this paper, we decompose the process of empirical software fault study into three key phases: (1) research objective definition, (2) data preparation, and (3) fault analysis, and we conduct an initial exploration study of applying Large Language Models (LLMs) for fault analysis of open-source software. Specifically, we perform the evaluation on 3,829 software faults drawn from a high-quality empirical study. Our results show that LLMs can substantially improve efficiency in fault analysis, with an average processing time of about two hours, compared to the weeks of manual effort typically required. We conclude by outlining a detailed research plan that highlights both the potential of LLMs for advancing empirical fault studies and the open challenges that required be addressed to achieve fully automated, end-to-end software fault analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v1">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v2">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 79 pages, 27 figures, 31 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04986v1">Observing Without Doing: Pseudo-Apprenticeship Patterns in Student LLM Use</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT have quickly become part of student programmers' toolkits, whether allowed by instructors or not. This paper examines how introductory programming (CS1) students integrate LLMs into their problem-solving processes. We conducted a mixed-methods study with 14 undergraduates completing three programming tasks while thinking aloud and permitted to access any resources they choose. The tasks varied in open-endedness and familiarity to the participants and were followed by surveys and interviews. We find that students frequently adopt a pattern we call pseudo-apprenticeship, where students engage attentively with expert-level solutions provided by LLMs but fail to participate in the stages of cognitive apprenticeship that promote independent problem-solving. This pattern was augmented by disconnects between students' intentions, actions, and self-perceived behavior when using LLMs. We offer design and instructional interventions for promoting learning and addressing the patterns of dependent AI use observed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v3">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v2">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04950v1">Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper)</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages, 3 tables; includes Limitations and Ethical Considerations sections; short paper under submission to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04919v1">Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v2">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Alignment of pretrained LLMs using instruction-based datasets is critical for creating fine-tuned models that reflect human preference. A growing number of alignment-based fine-tuning algorithms and benchmarks emerged recently, fueling the efforts on effective alignments of pre-trained LLMs to ensure helpful, harmless, and honest answers from both open-source and closed-source LLMs. This paper tackles this problem by developing an alignment fusion approach, coined as $H^3$Fusion, with three unique characteristics. First, $H^3$Fusion ensembles multiple individually aligned LLMs to create a final fine-tuned alignment model with enhanced capabilities beyond those of individual models, delivering robust alignment through promoting helpful, harmless, honest fusion. Second, $H^3$Fusion leverages the mixture-of-experts (MoE) methodology in two steps. We first freeze the multi-head attention weights of each individual model while tuning the FFN layer during alignment fusion. Then we merge the aligned model weights with an expert router according to the type of input instruction and dynamically select a subset of experts that are best suited for producing the output response. Finally, we boost the performance of the resulting $H^3$3Fusion model by introducing gating loss and regularization terms. The former penalizes the selection errors of the expert-router, and the latter mediates the expert weights drifting during fine-tuning and dynamically adjusts the fusion behavior of the resulting model by canalizing the activations on the experts. Extensive evaluations on three benchmark datasets show that $H^3$3Fusion is more helpful, less harmful, and more honest from two aspects: it outperforms each individually aligned model by $11.37\%$, and it provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by $13.77\%$. Code is available at github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04891v1">SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04885v1">RL Is a Hammer and LLMs Are Nails: A Simple Reinforcement Learning Recipe for Strong Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Prompt injection poses a serious threat to the reliability and safety of LLM agents. Recent defenses against prompt injection, such as Instruction Hierarchy and SecAlign, have shown notable robustness against static attacks. However, to more thoroughly evaluate the robustness of these defenses, it is arguably necessary to employ strong attacks such as automated red-teaming. To this end, we introduce RL-Hammer, a simple recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks via reinforcement learning. RL-Hammer requires no warm-up data and can be trained entirely from scratch. To achieve high ASRs against industrial-level models with defenses, we propose a set of practical techniques that enable highly effective, universal attacks. Using this pipeline, RL-Hammer reaches a 98% ASR against GPT-4o and a $72\%$ ASR against GPT-5 with the Instruction Hierarchy defense. We further discuss the challenge of achieving high diversity in attacks, highlighting how attacker models tend to reward-hack diversity objectives. Finally, we show that RL-Hammer can evade multiple prompt injection detectors. We hope our work advances automatic red-teaming and motivates the development of stronger, more principled defenses. Code is available at https://github.com/facebookresearch/rl-injector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04860v1">Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark Qwen3-8B and Llama-3.1-8B-Instruct. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide only fragile defenses against alignment tipping. Together, these findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at https://github.com/aiming-lab/ATP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04851v1">LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      We introduce LEGOMem, a modular procedural memory framework for multi-agent large language model (LLM) systems in workflow automation. LEGOMem decomposes past task trajectories into reusable memory units and flexibly allocates them across orchestrators and task agents to support planning and execution. To explore the design space of memory in multi-agent systems, we use LEGOMem as a lens and conduct a systematic study of procedural memory in multi-agent systems, examining where memory should be placed, how it should be retrieved, and which agents benefit most. Experiments on the OfficeBench benchmark show that orchestrator memory is critical for effective task decomposition and delegation, while fine-grained agent memory improves execution accuracy. We find that even teams composed of smaller language models can benefit substantially from procedural memory, narrowing the performance gap with stronger agents by leveraging prior execution traces for more accurate planning and tool use. These results position LEGOMem as both a practical framework for memory-augmented agent systems and a research tool for understanding memory design in multi-agent workflow automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04848v1">Instability in Downstream Task Performance During LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      When training large language models (LLMs), it is common practice to track downstream task performance throughout the training process and select the checkpoint with the highest validation score. However, downstream metrics often exhibit substantial fluctuations, making it difficult to identify the checkpoint that truly represents the best-performing model. In this study, we empirically analyze the stability of downstream task performance in an LLM trained on diverse web-scale corpora. We find that task scores frequently fluctuate throughout training, both at the aggregate and example levels. To address this instability, we investigate two post-hoc checkpoint integration methods: checkpoint averaging and ensemble, motivated by the hypothesis that aggregating neighboring checkpoints can reduce performance volatility. We demonstrate both empirically and theoretically that these methods improve downstream performance stability without requiring any changes to the training procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11723v2">Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Updated version with additional models and benchmark datasets. The experimental section has been expanded with new analyses, and minor corrections and clarifications have been made throughout the text
    </div>
    <details class="paper-abstract">
      Decoding strategies significantly influence the quality and diversity of the generated text in Large Language Models (LLMs), yet their impact on computational resources, particularly GPU energy consumption, is insufficiently studied. This paper investigates the relationship between text generation decoding techniques and energy efficiency, focusing on the trade-off between generation quality and GPU energy usage across diverse tasks and decoding configurations. By benchmarking multiple strategies across various tasks, including Translation, Math Problem Solving, Coding, and Open-ended text generation, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for energy consumption. Our findings show that the choice of decoding strategy can greatly impact GPU energy usage, even when it has a minimal effect on output quality. Different strategies also involve trade-offs between quality and energy efficiency, and no single decoding method is best in all cases across every metric. To the best of our knowledge, this is one of the first studies to examine decoding strategies in LLMs from the perspective of energy consumption, providing useful insights for building energy-efficient applications without compromising text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22777v4">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 October ARR
    </div>
    <details class="paper-abstract">
      Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03102v2">Semantic Similarity in Radiology Reports via LLMs and NER</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: https://github.com/otmive/llama_reports
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04796v1">RevMine: An LLM-Assisted Tool for Code Review Mining and Analysis Across Git Platforms</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Empirical research on code review processes is increasingly central to understanding software quality and collaboration. However, collecting and analyzing review data remains a time-consuming and technically intensive task. Most researchers follow similar workflows - writing ad hoc scripts to extract, filter, and analyze review data from platforms like GitHub and GitLab. This paper introduces RevMine, a conceptual tool that streamlines the entire code review mining pipeline using large language models (LLMs). RevMine guides users through authentication, endpoint discovery, and natural language-driven data collection, significantly reducing the need for manual scripting. After retrieving review data, it supports both quantitative and qualitative analysis based on user-defined filters or LLM-inferred patterns. This poster outlines the tool's architecture, use cases, and research potential. By lowering the barrier to entry, RevMine aims to democratize code review mining and enable a broader range of empirical software engineering studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22075v2">COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04774v1">Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04773v1">Distribution Preference Optimization: A Fine-grained Perspective for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) demonstrate remarkable capabilities learned from vast corpora, concerns regarding data privacy and safety are receiving increasing attention. LLM unlearning, which aims to remove the influence of specific data while preserving overall model utility, is becoming an important research area. One of the mainstream unlearning classes is optimization-based methods, which achieve forgetting directly through fine-tuning, exemplified by Negative Preference Optimization (NPO). However, NPO's effectiveness is limited by its inherent lack of explicit positive preference signals. Attempts to introduce such signals by constructing preferred responses often necessitate domain-specific knowledge or well-designed prompts, fundamentally restricting their generalizability. In this paper, we shift the focus to the distribution-level, directly targeting the next-token probability distribution instead of entire responses, and derive a novel unlearning algorithm termed \textbf{Di}stribution \textbf{P}reference \textbf{O}ptimization (DiPO). We show that the requisite preference distribution pairs for DiPO, which are distributions over the model's output tokens, can be constructed by selectively amplifying or suppressing the model's high-confidence output logits, thereby effectively overcoming NPO's limitations. We theoretically prove the consistency of DiPO's loss function with the desired unlearning direction. Extensive experiments demonstrate that DiPO achieves a strong trade-off between model utility and forget quality. Notably, DiPO attains the highest forget quality on the TOFU benchmark, and maintains leading scalability and sustainability in utility preservation on the MUSE benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01238v2">Silent Tokens, Loud Effects: Padding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Padding tokens are widely used in large language models (LLMs) to equalize sequence lengths during batched inference. While they should be fully masked, implementation errors can cause them to influence computation, and the extent of this influence is not well understood. We systematically study this effect across three open-source model families (Llama, Gemma, Qwen), inserting controlled amounts of padding and evaluating outcomes along four axes: activations, generation quality, bias, and safety. Even small amounts of padding shift hidden representations, degrade quality in smaller models, alter bias in unpredictable ways, and weaken safety guardrails. These findings demonstrate that padding is not a harmless detail but a robustness risk that must be carefully handled in deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04767v1">ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Project Page: https://parallelbench.github.io
    </div>
    <details class="paper-abstract">
      While most autoregressive LLMs are constrained to one-by-one decoding, diffusion LLMs (dLLMs) have attracted growing interest for their potential to dramatically accelerate inference through parallel decoding. Despite this promise, the conditional independence assumption in dLLMs causes parallel decoding to ignore token dependencies, inevitably degrading generation quality when these dependencies are strong. However, existing works largely overlook these inherent challenges, and evaluations on standard benchmarks (e.g., math and coding) are not sufficient to capture the quality degradation caused by parallel decoding. To address this gap, we first provide an information-theoretic analysis of parallel decoding. We then conduct case studies on analytically tractable synthetic list operations from both data distribution and decoding strategy perspectives, offering quantitative insights that highlight the fundamental limitations of parallel decoding. Building on these insights, we propose ParallelBench, the first benchmark specifically designed for dLLMs, featuring realistic tasks that are trivial for humans and autoregressive LLMs yet exceptionally challenging for dLLMs under parallel decoding. Using ParallelBench, we systematically analyze both dLLMs and autoregressive LLMs, revealing that: (i) dLLMs under parallel decoding can suffer dramatic quality degradation in real-world scenarios, and (ii) current parallel decoding strategies struggle to adapt their degree of parallelism based on task difficulty, thus failing to achieve meaningful speedup without compromising quality. Our findings underscore the pressing need for innovative decoding methods that can overcome the current speed-quality trade-off. We release our benchmark to help accelerate the development of truly efficient dLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01667v3">Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 23pages, 4 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks in Luxembourgish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04749v1">LLM-Based Information Extraction to Support Scientific Literature Research and Publication Workflows</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 This PDF is the author-prepared camera-ready version corresponding to the accepted manuscript and supersedes the submitted version that was inadvertently published as the version of record
    </div>
    <details class="paper-abstract">
      The increasing volume of scholarly publications requires advanced tools for efficient knowledge discovery and management. This paper introduces ongoing work on a system using Large Language Models (LLMs) for the semantic extraction of key concepts from scientific documents. Our research, conducted within the German National Research Data Infrastructure for and with Computer Science (NFDIxCS) project, seeks to support FAIR (Findable, Accessible, Interoperable, and Reusable) principles in scientific publishing. We outline our explorative work, which uses in-context learning with various LLMs to extract concepts from papers, initially focusing on the Business Process Management (BPM) domain. A key advantage of this approach is its potential for rapid domain adaptation, often requiring few or even zero examples to define extraction targets for new scientific fields. We conducted technical evaluations to compare the performance of commercial and open-source LLMs and created an online demo application to collect feedback from an initial user-study. Additionally, we gathered insights from the computer science research community through user stories collected during a dedicated workshop, actively guiding the ongoing development of our future services. These services aim to support structured literature reviews, concept-based information retrieval, and integration of extracted knowledge into existing knowledge graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07644v2">FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 v2, Project page: https://OldDelorean.github.io/FloorplanQA/
    </div>
    <details class="paper-abstract">
      We introduce FloorplanQA, a diagnostic benchmark for evaluating spatial reasoning in large-language models (LLMs). FloorplanQA is grounded in structured representations of indoor scenes, such as (e.g., kitchens, living rooms, bedrooms, bathrooms, and others), encoded symbolically in JSON or XML layouts. The benchmark covers core spatial tasks, including distance measurement, visibility, path finding, and object placement within constrained spaces. Our results across a variety of frontier open-source and commercial LLMs reveal that while models may succeed in shallow queries, they often fail to respect physical constraints, preserve spatial coherence, though they remain mostly robust to small spatial perturbations. FloorplanQA uncovers a blind spot in today's LLMs: inconsistent reasoning about indoor layouts. We hope this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04721v1">BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04717v1">JSON Whisperer: Efficient JSON Editing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can modify JSON documents through natural language commands, but current approaches regenerate entire structures for each edit, resulting in computational inefficiency. We present JSON Whisperer, a framework that enables LLMs to generate RFC 6902 diff patches-expressing only the necessary modifications-rather than complete documents. We identify two key challenges in patch-based editing: (1) LLMs often miss related updates when generating isolated patches, and (2) array manipulations require tracking index shifts across operations, which LLMs handle poorly. To address these issues, we introduce EASE (Explicitly Addressed Sequence Encoding), which transforms arrays into dictionaries with stable keys, eliminating index arithmetic complexities. Our evaluation shows that patch generation with EASE reduces token usage by 31% while maintaining edit quality within 5% of full regeneration with particular gains for complex instructions and list manipulations. The dataset is available at: https://github.com/emnlp2025/JSON-Whisperer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04695v1">Beyond Outcome Reward: Decoupling Search and Answering Improves LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Enabling large language models (LLMs) to utilize search tools offers a promising path to overcoming fundamental limitations such as knowledge cutoffs and hallucinations. Recent work has explored reinforcement learning (RL) for training search-augmented agents that interleave reasoning and retrieval before answering. These approaches usually rely on outcome-based rewards (e.g., exact match), implicitly assuming that optimizing for final answers will also yield effective intermediate search behaviors. Our analysis challenges this assumption: we uncover multiple systematic deficiencies in search that arise under outcome-only training and ultimately degrade final answer quality, including failure to invoke tools, invalid queries, and redundant searches. To address these shortcomings, we introduce DeSA (Decoupling Search-and-Answering), a simple two-stage training framework that explicitly separates search optimization from answer generation. In Stage 1, agents are trained to improve search effectiveness with retrieval recall-based rewards. In Stage 2, outcome rewards are employed to optimize final answer generation. Across seven QA benchmarks, DeSA-trained agents consistently improve search behaviors, delivering substantially higher search recall and answer accuracy than outcome-only baselines. Notably, DeSA outperforms single-stage training approaches that simultaneously optimize recall and outcome rewards, underscoring the necessity of explicitly decoupling the two objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v4">SALAD: Systematic Assessment of Machine Unlearning on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04641v1">Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 17 pages, 7 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04637v1">Social Agent: Mastering Dyadic Nonverbal Behavior Generation via Conversational LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 SIGGRAPH ASIA 2025 (Conference Track); Project page: https://pku-mocca.github.io/Social-Agent-Page/
    </div>
    <details class="paper-abstract">
      We present Social Agent, a novel framework for synthesizing realistic and contextually appropriate co-speech nonverbal behaviors in dyadic conversations. In this framework, we develop an agentic system driven by a Large Language Model (LLM) to direct the conversation flow and determine appropriate interactive behaviors for both participants. Additionally, we propose a novel dual-person gesture generation model based on an auto-regressive diffusion model, which synthesizes coordinated motions from speech signals. The output of the agentic system is translated into high-level guidance for the gesture generator, resulting in realistic movement at both the behavioral and motion levels. Furthermore, the agentic system periodically examines the movements of interlocutors and infers their intentions, forming a continuous feedback loop that enables dynamic and responsive interactions between the two participants. User studies and quantitative evaluations show that our model significantly improves the quality of dyadic interactions, producing natural, synchronized nonverbal behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02369v2">Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04633v1">Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The unjudged document problem, where pooled test collections have incomplete relevance judgments for evaluating new retrieval systems, is a key obstacle to the reusability of test collections in information retrieval. While the de facto standard to deal with the problem is to treat unjudged documents as non-relevant, many alternatives have been proposed, including the use of large language models (LLMs) as a relevance judge (LLM-as-a-judge). However, this has been criticized as circular, since the same LLM can be used as a judge and as a ranker at the same time. We propose to train topic-specific relevance classifiers instead: By finetuning monoT5 with independent LoRA weight adaptation on the judgments of a single assessor for a single topic's pool, we align it to that assessor's notion of relevance for the topic. The system rankings obtained through our classifier's relevance judgments achieve a Spearmans' $\rho$ correlation of $>0.95$ with ground truth system rankings. As little as 128 initial human judgments per topic suffice to improve the comparability of models, compared to treating unjudged documents as non-relevant, while achieving more reliability than existing LLM-as-a-judge approaches. Topic-specific relevance classifiers thus are a lightweight and straightforward way to tackle the unjudged document problem, while maintaining human judgments as the gold standard for retrieval evaluation. Code, models, and data are made openly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01617v2">AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted by EMNLP-2025 Industrial Track
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04607v1">A Case for Declarative LLM-friendly Interfaces for Improved Efficiency of Computer-Use Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Computer-use agents (CUAs) powered by large language models (LLMs) have emerged as a promising approach to automating computer tasks, yet they struggle with graphical user interfaces (GUIs). GUIs, designed for humans, force LLMs to decompose high-level goals into lengthy, error-prone sequences of fine-grained actions, resulting in low success rates and an excessive number of LLM calls. We propose Goal-Oriented Interface (GOI), a novel abstraction that transforms existing GUIs into three declarative primitives: access, state, and observation, which are better suited for LLMs. Our key idea is policy-mechanism separation: LLMs focus on high-level semantic planning (policy) while GOI handles low-level navigation and interaction (mechanism). GOI does not require modifying the application source code or relying on application programming interfaces (APIs). We evaluate GOI with Microsoft Office Suite (Word, PowerPoint, Excel) on Windows. Compared to a leading GUI-based agent baseline, GOI improves task success rates by 67% and reduces interaction steps by 43.5%. Notably, GOI completes over 61% of successful tasks with a single LLM call.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10364v3">Can We Infer Confidential Properties of Training Data from LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04581v1">Can LLMs Detect Ambiguous Plural Reference? An Analysis of Split-Antecedent and Mereological Reference</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Our goal is to study how LLMs represent and interpret plural reference in ambiguous and unambiguous contexts. We ask the following research questions: (1) Do LLMs exhibit human-like preferences in representing plural reference? (2) Are LLMs able to detect ambiguity in plural anaphoric expressions and identify possible referents? To address these questions, we design a set of experiments, examining pronoun production using next-token prediction tasks, pronoun interpretation, and ambiguity detection using different prompting strategies. We then assess how comparable LLMs are to humans in formulating and interpreting plural reference. We find that LLMs are sometimes aware of possible referents of ambiguous pronouns. However, they do not always follow human reference when choosing between interpretations, especially when the possible interpretation is not explicitly mentioned. In addition, they struggle to identify ambiguity without direct instruction. Our findings also reveal inconsistencies in the results across different types of experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v1">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04567v1">GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are powerful tools for precessing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach.
    </details>
</div>
