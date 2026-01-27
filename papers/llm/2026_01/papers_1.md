# llm - 2026_01

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18785v1">Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Extended abstract presented at the 2025 Wordplay Workshop at EMNLP
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has enabled a new paradigm for bridging authorial intent and player agency in interactive narrative. We consider this paradigm through the example of Dramamancer, a system that uses an LLM to transform author-created story schemas into player-driven playthroughs. This extended abstract outlines some design techniques and evaluation considerations associated with this system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18777v1">PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at AAAI 2026 - Innovative Applications of AI (IAAI-26)
    </div>
    <details class="paper-abstract">
      Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18754v1">$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18753v1">HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Have been accepted by ICLR'26
    </div>
    <details class="paper-abstract">
      The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: data-driven hallucinations and reasoning-driven hallucinations. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the Hallucination Risk Bound, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce HalluGuard, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate HalluGuard on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02599v2">Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18731v1">One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and automatically provide individualized feedback. However, developing these models faces two critical challenges: the scarcity of feedback from individual users and the need for efficient adaptation to unseen users. We argue that addressing these constraints requires a paradigm shift from fitting data to learn user preferences to learn the process of preference adaptation. To realize this, we propose Meta Reward Modeling (MRM), which reformulates personalized reward modeling as a meta-learning problem. Specifically, we represent each user's reward model as a weighted combination of base reward functions, and optimize the initialization of these weights using a Model-Agnostic Meta-Learning (MAML)-style framework to support fast adaptation under limited feedback. To ensure robustness, we introduce the Robust Personalization Objective (RPO), which places greater emphasis on hard-to-learn users during meta optimization. Extensive experiments on personalized preference datasets validate that MRM enhances few-shot personalization, improves user robustness, and consistently outperforms baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18706v1">Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Rubrics are essential for evaluating open-ended LLM responses, especially in safety-critical domains such as healthcare. However, creating high-quality and domain-specific rubrics typically requires significant human expertise time and development cost, making rubric-based evaluation and training difficult to scale. In this work, we introduce Health-SCORE, a generalizable and scalable rubric-based training and evaluation framework that substantially reduces rubric development costs without sacrificing performance. We show that Health-SCORE provides two practical benefits beyond standalone evaluation: it can be used as a structured reward signal to guide reinforcement learning with safety-aware supervision, and it can be incorporated directly into prompts to improve response quality through in-context learning. Across open-ended healthcare tasks, Health-SCORE achieves evaluation quality comparable to human-created rubrics while significantly lowering development effort, making rubric-based evaluation and training more scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17853v3">CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which assesses whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 17%, and achieves up to 68.1% accuracy on the CiteME benchmark, approaching human-level performance (69.7%). It also enables the identification of alternative but valid citations and demonstrates generalization ability for cross-domain citation attribution.Our code is available at https://github.com/KathCYM/CiteGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12945v3">LLMPopcorn: Exploring LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted by ICASSP2026
    </div>
    <details class="paper-abstract">
      In an era where micro-videos dominate platforms like TikTok and YouTube, AI-generated content is nearing cinematic quality. The next frontier is using large language models (LLMs) to autonomously create viral micro-videos, a largely untapped potential that could shape the future of AI-driven content creation. To address this gap, this paper presents the first exploration of LLM-assisted popular micro-video generation (LLMPopcorn). We selected popcorn as the icon for this paper because it symbolizes leisure and entertainment, aligning with this study on leveraging LLMs as assistants for generating popular micro-videos that are often consumed during leisure time. Specifically, we empirically study the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? Exploring these questions, we show that advanced LLMs like DeepSeek-V3 can generate micro-videos with popularity rivaling human content. Prompt enhancement further boosts results, while benchmarking highlights DeepSeek-V3 and R1 for LLMs, and LTX-Video and HunyuanVideo for video generation. This work advances AI-assisted micro-video creation and opens new research directions. The code is publicly available at https://github.com/GAIR-Lab/LLMPopcorn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06822v2">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The advent of complex, interconnected long-horizon LLM systems has made it incredibly tricky to identify where and when these systems break down. Evaluation capabilities that currently exist today are limited in that they often focus on simple metrics, end-to-end outcomes, and are dependent on the perspectives of humans. In order to match the increasing complexity of these many component systems, evaluation frameworks must also be able to reason, probe, iterate, and understand the nuanced logic passing through these systems. In this paper, we present RAFFLES, an offline evaluation architecture that incorporates iterative reasoning. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically identify faults and a set of specialized Evaluators to assess the quality of the candidate faults as well as rationales of the Judge. We evaluated RAFFLES with several benchmarks - the Who&When dataset to identify step-level faults in multi-agent systems and the ReasonEval datasets to diagnose step-level mathematical reasoning errors. RAFFLES outperforms strong baselines, achieving an accuracy of over 20% and 50% on the Who&When Hand-Crafted and Algorithmically-Generated datasets, and over 80% on the ReasonEval datasets. These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15765v2">Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Data is a critical asset for training large language models (LLMs), alongside compute resources and skilled workers. While some training data is publicly available, substantial investment is required to generate proprietary datasets, such as human preference annotations or to curate new ones from existing sources. As larger datasets generally yield better model performance, two natural questions arise. First, how can data owners make informed decisions about curation strategies and data sources investment? Second, how can multiple data owners collaboratively pool their resources to train superior models while fairly distributing the benefits? This problem, data valuation, which is not specific to large language models, has been addressed by the machine learning community through the lens of cooperative game theory, with the Shapley value being the prevalent solution concept. However, computing Shapley values is notoriously expensive for data valuation, typically requiring numerous model retrainings, which can become prohibitive for large machine learning models. In this work, we demonstrate that this computational challenge is dramatically simplified for LLMs trained with Direct Preference Optimization (DPO). We show how the specific mathematical structure of DPO enables scalable Shapley value computation. We believe this observation unlocks many applications at the intersection of data valuation and large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18630v1">Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The escalating global mental health crisis, marked by persistent treatment gaps, availability, and a shortage of qualified therapists, positions Large Language Models (LLMs) as a promising avenue for scalable support. While LLMs offer potential for accessible emotional assistance, their reliability, therapeutic relevance, and alignment with human standards remain challenging to address. This paper introduces a human-grounded evaluation methodology designed to assess LLM generated responses in therapeutic dialogue. Our approach involved curating a dataset of 500 mental health conversations from datasets with real-world scenario questions and evaluating the responses generated by nine diverse LLMs, including closed source and open source models. More specifically, these responses were evaluated by two psychiatric trained experts, who independently rated each on a 5 point Likert scale across a comprehensive 6 attribute rubric. This rubric captures Cognitive Support and Affective Resonance, providing a multidimensional perspective on therapeutic quality. Our analysis reveals that LLMs provide strong cognitive reliability by producing safe, coherent, and clinically appropriate information, but they demonstrate unstable affective alignment. Although closed source models (e.g., GPT-4o) offer balanced therapeutic responses, open source models show greater variability and emotional flatness. We reveal a persistent cognitive-affective gap and highlight the need for failure aware, clinically grounded evaluation frameworks that prioritize relational sensitivity alongside informational accuracy in mental health oriented LLMs. We advocate for balanced evaluation protocols with human in the loop that center on therapeutic sensitivity and provide a framework to guide the responsible design and clinical oversight of mental health oriented conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18582v1">From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 9 pages, 4 figures, AAAI 2026 Bridge
    </div>
    <details class="paper-abstract">
      Personality detection aims to measure an individual's corresponding personality traits through their social media posts. The advancements in Large Language Models (LLMs) offer novel perspectives for personality detection tasks. Existing approaches enhance personality trait analysis by leveraging LLMs to extract semantic information from textual posts as prompts, followed by training classifiers for categorization. However, accurately classifying personality traits remains challenging due to the inherent complexity of human personality and subtle inter-trait distinctions. Moreover, prompt-based methods often exhibit excessive dependency on expert-crafted knowledge without autonomous pattern-learning capacity. To address these limitations, we view personality detection as a ranking task rather than a classification and propose a corresponding reinforcement learning training paradigm. First, we employ supervised fine-tuning (SFT) to establish personality trait ranking capabilities while enforcing standardized output formats, creating a robust initialization. Subsequently, we introduce Group Relative Policy Optimization (GRPO) with a specialized ranking-based reward function. Unlike verification tasks with definitive solutions, personality assessment involves subjective interpretations and blurred boundaries between trait categories. Our reward function explicitly addresses this challenge by training LLMs to learn optimal answer rankings. Comprehensive experiments have demonstrated that our method achieves state-of-the-art performance across multiple personality detection benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18572v1">One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfair outcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to study bias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variations (robustness) and the rarity of some cues in real interactions (external validity). We compare six commonly used persona cues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce substantial variance in responses across personas. We therefore caution against claims from a single persona cue and recommend future personalization research to evaluate multiple externally valid cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18563v1">An LLM-Agent-Based Framework for Age of Information Optimization in Heterogeneous Random Access Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      With the rapid expansion of the Internet of Things (IoT) and heterogeneous wireless networks, the Age of Information (AoI) has emerged as a critical metric for evaluating the performance of real-time and personalized systems. While AoI-based random access is essential for next-generation applications such as the low-altitude economy and indoor service robots, existing strategies, ranging from rule-based protocols to learning-based methods, face critical challenges, including idealized model assumptions, slow convergence, and poor generalization. In this article, we propose Reflex-Core, a novel Large Language Model (LLM) agent-based framework for AoI-driven random access in heterogeneous networks. By devising an "Observe-Reflect-Decide-Execute" closed-loop mechanism, this framework integrates Supervised Fine-Tuning (SFT) and Proximal Policy Optimization (PPO) to enable optimal, autonomous access control. Based on the Reflex-Core framework, we develop a Reflexive Multiple Access (RMA) protocol and a priority-based RMA variant for intelligent access control under different heterogeneous network settings. Experimental results demonstrate that in the investigated scenarios, the RMA protocol achieves up to a 14.9% reduction in average AoI compared with existing baselines, while the priority-based version improves the convergence rate by approximately 20%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18552v1">Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs are increasingly embedded in everyday decision-making, yet their outputs can encode subtle, unintended behaviours that shape user beliefs and actions. We refer to these covert, goal-directed behaviours as hidden intentions, which may arise from training and optimisation artefacts, or be deliberately induced by an adversarial developer, yet remain difficult to detect in practice. We introduce a taxonomy of ten categories of hidden intentions, grounded in social science research and organised by intent, mechanism, context, and impact, shifting attention from surface-level behaviours to design-level strategies of influence. We show how hidden intentions can be easily induced in controlled models, providing both testbeds for evaluation and demonstrations of potential misuse. We systematically assess detection methods, including reasoning and non-reasoning LLM judges, and find that detection collapses in realistic open-world settings, particularly under low-prevalence conditions, where false positives overwhelm precision and false negatives conceal true risks. Stress tests on precision-prevalence and precision-FNR trade-offs reveal why auditing fails without vanishingly small false positive rates or strong priors on manipulation types. Finally, a qualitative case study shows that all ten categories manifest in deployed, state-of-the-art LLMs, emphasising the urgent need for robust frameworks. Our work provides the first systematic analysis of detectability failures of hidden intentions in LLMs under open-world settings, offering a foundation for understanding, inducing, and stress-testing such behaviours, and establishing a flexible taxonomy for anticipating evolving threats and informing governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18052v2">The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 13 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed to simulate human collective behaviors, yet the methodological rigor of these "AI societies" remains under-explored. Through a systematic audit of 42 recent studies, we identify six pervasive flaws-spanning agent profiles, interaction, memory, control, unawareness, and realism (PIMMUR). Our analysis reveals that 90.7% of studies violate at least one principle, undermining simulation validity. We demonstrate that frontier LLMs correctly identify the underlying social experiment in 47.6% of cases, while 65.3% of prompts exert excessive control that pre-determines outcomes. By reproducing five representative experiments (e.g., telephone game), we show that reported collective phenomena often vanish or reverse when PIMMUR principles are enforced, suggesting that many "emergent" behaviors are methodological artifacts rather than genuine social dynamics. Our findings suggest that current AI simulations may capture model-specific biases rather than universal human social behaviors, raising critical concerns about the use of LLMs as scientific proxies for human society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18510v1">Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at https://github.com/liushiliushi/JitRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18492v1">DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18483v1">Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at EACL main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer strong generative capabilities, but many applications require explicit and \textit{fine-grained} control over specific textual concepts, such as humor, persuasiveness, or formality. Prior approaches in prompting and representation engineering can provide coarse or single-attribute control, but systematic evaluation of multi-attribute settings remains limited. We introduce an evaluation framework for fine-grained controllability for both single- and dual-concept scenarios, focusing on linguistically distinct concept pairs (e.g., persuasiveness vs.~humor). Surprisingly, across multiple LLMs and generative tasks, we find that performance often drops in the dual-concept setting, even though the chosen concepts should in principle be separable. This reveals a fundamental limitation of naive prompting-based control: models struggle with compositionality even when concepts are intuitively independent. Our framework provides systematic evidence of this gap and offers a principled approach for measuring the ability of future methods for multi-concept control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18457v1">Token-level Collaborative Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures, 7 tables, WWW 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong potential for generative recommendation by leveraging rich semantic knowledge. However, existing LLM-based recommender systems struggle to effectively incorporate collaborative filtering (CF) signals, due to a fundamental mismatch between item-level preference modeling in CF and token-level next-token prediction (NTP) optimization in LLMs. Prior approaches typically treat CF as contextual hints or representation bias, and resort to multi-stage training to reduce behavioral semantic space discrepancies, leaving CF unable to explicitly regulate LLM generation. In this work, we propose Token-level Collaborative Alignment for Recommendation (TCA4Rec), a model-agnostic and plug-and-play framework that establishes an explicit optimization-level interface between CF supervision and LLM generation. TCA4Rec consists of (i) Collaborative Tokenizer, which projects raw item-level CF logits into token-level distributions aligned with the LLM token space, and (ii) Soft Label Alignment, which integrates these CF-informed distributions with one-hot supervision to optimize a soft NTP objective. This design preserves the generative nature of LLM training while enabling collaborative alignment with essential user preference of CF models. We highlight TCA4Rec is compatible with arbitrary traditional CF models and generalizes across a wide range of decoder-based LLM recommender architectures. Moreover, it provides an explicit mechanism to balance behavioral alignment and semantic fluency, yielding generative recommendations that are both accurate and controllable. Extensive experiments demonstrate that TCA4Rec consistently improves recommendation performance across a broad spectrum of CF models and LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v3">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and a lack of customized modeling frameworks. This paper proposes an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategies, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation. To handle the scalability issues inherent in large-scale P2P networks, a differential attention-based critic network is introduced to efficiently extract key interaction features and enhance convergence. Experimental results demonstrate that LLM-generated strategies effectively substitute human experts. The proposed imitative expert MARL algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baseline algorithms, while maintaining robust stability. This paper provides an effective solution for the real-time decision-making of the P2P electricity market by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20984v2">Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 NeurIPS 2025 Poster
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities but typically require extensive computational resources and memory for inference. Post-training quantization (PTQ) can effectively reduce these demands by storing weights in lower bit-width formats. However, standard uniform quantization often leads to notable performance degradation, particularly in low-bit scenarios. In this work, we introduce a Grouped Lattice Vector Quantization (GLVQ) framework that assigns each group of weights a customized lattice codebook, defined by a learnable generation matrix. To address the non-differentiability of the quantization process, we adopt Babai rounding to approximate nearest-lattice-point search during training, which enables stable optimization of the generation matrices. Once trained, decoding reduces to a simple matrix-vector multiplication, yielding an efficient and practical quantization pipeline. Experiments on multiple benchmarks show that our approach achieves a better trade-off between model size and accuracy compared to existing post-training quantization baselines, highlighting its effectiveness in deploying large models under stringent resource constraints. Our source code is available on GitHub repository: https://github.com/xzhang9308/GLVQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16495v2">Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Revised
    </div>
    <details class="paper-abstract">
      Efficient parallelism is necessary for achieving low-latency, high-throughput inference with large language models (LLMs). Tensor parallelism (TP) is the state-of-the-art method for reducing LLM response latency, however GPU communications reduces combined token throughput. On the other hand, data parallelism (DP) obtains a higher throughput yet is slow in response latency. Best of both worlds does not exist, and it is not possible to combine TP and DP because of the KV cache variance across the parallelisms. We notice Sequence Parallelism (SP - Ulysses in training) has similar properties as DP but with KV cache invariance. We adapt SP to inference, and combine it with TP to get the best of both worlds. Our solution: Shift Parallelism. Shift Parallelism dynamically switches across TP and SP, and minimizes latency in low traffic without losing throughput in high traffic. The efficient GPU communications of Shift Parallelism yields up to i) 1.51x faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution. We evaluate Shift Parallelism with real-world production traces with dynamic traffic patterns as well as synthetic benchmarking patterns across models, context sizes, and arrival rates. All results affirm the same: Shift Parallelism has a better the latency vs. throughput tradeoff than TP or DP, and hence obtains low latency without degrading throughput in dynamic workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08653v2">Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18375v1">Hierarchical Text Classification with LLM-Refined Taxonomies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Hierarchical text classification (HTC) depends on taxonomies that organize labels into structured hierarchies. However, many real-world taxonomies introduce ambiguities, such as identical leaf names under similar parent nodes, which prevent language models (LMs) from learning clear decision boundaries. In this paper, we present TaxMorph, a framework that uses large language models (LLMs) to transform entire taxonomies through operations such as renaming, merging, splitting, and reordering. Unlike prior work, our method revises the full hierarchy to better match the semantics encoded by LMs. Experiments across three HTC benchmarks show that LLM-refined taxonomies consistently outperform human-curated ones in various settings up to +2.9pp. in F1. To better understand these improvements, we compare how well LMs can assign leaf nodes to parent nodes and vice versa across human-curated and LLM-refined taxonomies. We find that human-curated taxonomies lead to more easily separable clusters in embedding space. However, the LLM-refined taxonomies align more closely with the model's actual confusion patterns during classification. In other words, even though they are harder to separate, they better reflect the model's inductive biases. These findings suggest that LLM-guided refinement creates taxonomies that are more compatible with how models learn, improving HTC performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18350v1">When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18334v1">Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into clinical workflows, their tendency for sycophancy, prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the Adjusted Sycophancy Score, a novel metric that isolates alignment bias by accounting for stochastic model instability, or "confusability". Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized "Thinking" models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18315v1">CovertComBench: The First Domain-Specific Testbed for LLMs in Wireless Covert Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 6pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wireless networks presents significant potential for automating system design. However, unlike conventional throughput maximization, Covert Communication (CC) requires optimizing transmission utility under strict detection-theoretic constraints, such as Kullback-Leibler divergence limits. Existing benchmarks primarily focus on general reasoning or standard communication tasks and do not adequately evaluate the ability of LLMs to satisfy these rigorous security constraints. To address this limitation, we introduce CovertComBench, a unified benchmark designed to assess LLM capabilities across the CC pipeline, encompassing conceptual understanding (MCQs), optimization derivation (ODQs), and code generation (CGQs). Furthermore, we analyze the reliability of automated scoring within a detection-theoretic ``LLM-as-Judge'' framework. Extensive evaluations across state-of-the-art models reveal a significant performance discrepancy. While LLMs achieve high accuracy in conceptual identification (81%) and code implementation (83%), their performance in the higher-order mathematical derivations necessary for security guarantees ranges between 18% and 55%. This limitation indicates that current LLMs serve better as implementation assistants rather than autonomous solvers for security-constrained optimization. These findings suggest that future research should focus on external tool augmentation to build trustworthy wireless AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18306v1">Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted to EACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Quantization is an effective technique for reducing the storage footprint and computational costs of Large Language Models (LLMs), but it often results in performance degradation. Existing post-training quantization methods typically use small, English-only calibration sets; however, their impact on multilingual models remains underexplored. We systematically evaluate eight calibration settings (five single-language and three multilingual mixes) on two quantizers (GPTQ, AWQ) on data from 10 languages. Our findings reveal a consistent trend: non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines. Specifically, we observe notable average perplexity gains across both quantizers on Llama3.1 8B and Qwen2.5 7B, with multilingual mixes achieving the largest overall reductions of up to 3.52 points in perplexity. Furthermore, our analysis indicates that tailoring calibration sets to the evaluation language yields the largest improvements for individual languages, underscoring the importance of linguistic alignment. We also identify specific failure cases where certain language-quantizer combinations degrade performance, which we trace to differences in activation range distributions across languages. These results highlight that static one-size-fits-all calibration is suboptimal and that tailoring calibration data, both in language and diversity, plays a crucial role in robustly quantizing multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01289v2">OntoMetric: An Ontology-Driven LLM-Assisted Framework for Automated ESG Metric Knowledge Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Environmental, Social, and Governance (ESG) metric knowledge is inherently structured, connecting industries, reporting frameworks, metric categories, metrics, and calculation models through compositional dependencies, yet in practice this structure remains embedded implicitly in regulatory documents such as SASB, TCFD, and IFRS S2 and rarely exists as an explicit, governed, or machine-actionable artefact. Existing ESG ontologies define formal schemas but do not address scalable population and governance from authoritative regulatory sources, while unconstrained large language model (LLM) extraction frequently produces semantically incorrect entities, hallucinated relationships, and structurally invalid graphs. OntoMetric is an ontology-guided framework for the automated construction and governance of ESG metric knowledge graphs from regulatory documents that operationalises the ESG Metric Knowledge Graph (ESGMKG) ontology as a first-class constraint embedded directly into the extraction and population process. The framework integrates structure-aware segmentation, ontology-constrained LLM extraction enriched with semantic fields and deterministic identifiers, and two-phase validation combining semantic type verification with rule-based schema checking, while preserving segment-level and page-level provenance to ensure traceability to regulatory source text. Evaluation on five ESG regulatory standards shows that ontology-guided extraction achieves 65-90 percent semantic accuracy and over 80 percent schema compliance, compared with 3-10 percent for unconstrained baseline extraction, and yields stable cost efficiency with a cost per validated entity of 0.01-0.02 USD and a 48 times efficiency improvement over baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18292v1">TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18282v1">Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15727v2">Towards Automated Kernel Generation in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18255v1">Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Continual learning in Large Language Models (LLMs) faces the critical challenge of balancing stability (retaining old knowledge) and plasticity (learning new tasks). While Experience Replay (ER) is a standard countermeasure against catastrophic forgetting, its impact across diverse capabilities remains underexplored. In this work, we uncover a critical dichotomy in ER's behavior: while it induces positive backward transfer on robust, unstructured tasks (e.g., boosting performance on previous NLP classification tasks through repeated rehearsal), it causes severe negative transfer on fragile, structured domains like code generation (e.g., a significant relative drop in coding accuracy). This reveals that ER trades structural integrity for broad consolidation. To address this dilemma, we propose \textbf{Orthogonal Subspace Wake-up (OSW)}. OSW identifies essential parameter subspaces of previous tasks via a brief "wake-up" phase and enforces orthogonal updates for new tasks, providing a mathematically grounded "safety guarantee" for established knowledge structures. Empirical results across a diverse four-task sequence demonstrate that OSW uniquely succeeds in preserving fragile coding abilities where Replay fails, while simultaneously maintaining high plasticity for novel tasks. Our findings emphasize the necessity of evaluating structural safety alongside average retention in LLM continual learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18253v1">BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 This is a pre-print
    </div>
    <details class="paper-abstract">
      Accurate evaluation of user satisfaction is critical for iterative development of conversational AI. However, for open-ended assistants, traditional A/B testing lacks reliable metrics: explicit feedback is sparse, while implicit metrics are ambiguous. To bridge this gap, we introduce BoRP (Bootstrapped Regression Probing), a scalable framework for high-fidelity satisfaction evaluation. Unlike generative approaches, BoRP leverages the geometric properties of LLM latent space. It employs a polarization-index-based bootstrapping mechanism to automate rubric generation and utilizes Partial Least Squares (PLS) to map hidden states to continuous scores. Experiments on industrial datasets show that BoRP (Qwen3-8B/14B) significantly outperforms generative baselines (even Qwen3-Max) in alignment with human judgments. Furthermore, BoRP reduces inference costs by orders of magnitude, enabling full-scale monitoring and highly sensitive A/B testing via CUPED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18241v1">TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at the 9th Workshop on Validation, Analysis and Evolution of Software Tests (VST 2026), co-located with the the 33rd IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2026)
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown promise in software engineering, their application to unit testing remains largely confined to isolated test generation or oracle prediction, neglecting the broader challenge of test suite maintenance. We introduce TAM-Eval (Test Automated Maintenance Evaluation), a framework and benchmark designed to evaluate model performance across three core test maintenance scenarios: creation, repair, and updating of test suites. Unlike prior work limited to function-level tasks, TAM-Eval operates at the test file level, while maintaining access to full repository context during isolated evaluation, better reflecting real-world maintenance workflows. Our benchmark comprises 1,539 automatically extracted and validated scenarios from Python, Java, and Go projects. TAM-Eval supports system-agnostic evaluation of both raw LLMs and agentic workflows, using a reference-free protocol based on test suite pass rate, code coverage, and mutation testing. Empirical results indicate that state-of-the-art LLMs have limited capabilities in realistic test maintenance processes and yield only marginal improvements in test effectiveness. We release TAM-Eval as an open-source framework to support future research in automated software testing. Our data and code are publicly available at https://github.com/trndcenter/TAM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00847v4">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 To appear in WWW 2026; 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $ε\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-ε}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18225v1">ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly deployed in e-commerce shopping. To perform thorough, user-tailored product searches, agents should interpret personal preferences, engage in multi-turn dialogues, and ultimately retrieve and discriminate among highly similar products. However, existing research has yet to provide a unified simulation environment that consistently captures all of these aspects, and always focuses solely on evaluation benchmarks without training support. In this paper, we introduce ShopSimulator, a large-scale and challenging Chinese shopping environment. Leveraging ShopSimulator, we evaluate LLMs across diverse scenarios, finding that even the best-performing models achieve less than 40% full-success rate. Error analysis reveals that agents struggle with deep search and product selection in long trajectories, fail to balance the use of personalization cues, and to effectively engage with users. Further training exploration provides practical guidance for overcoming these weaknesses, with the combination of supervised fine-tuning (SFT) and reinforcement learning (RL) yielding significant performance improvements. Code and data will be released at https://github.com/ShopAgent-Team/ShopSimulator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16781v2">Persuasion Tokens for Editing Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at EACL Main 2026
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18220v1">LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. The checkpoint and inference code will be released later.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18217v1">Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10132v2">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and prediction across different domains. Yet, their ability to infer temporal regularities from structured behavioral data remains underexplored. This paper presents a systematic study investigating whether LLMs can predict time intervals between recurring user actions, such as repeated purchases, and how different levels of contextual information shape their predictive behavior. Using a simple but representative repurchase scenario, we benchmark state-of-the-art LLMs in zero-shot settings against both statistical and machine-learning models. Two key findings emerge. First, while LLMs surpass lightweight statistical baselines, they consistently underperform dedicated machine-learning models, showing their limited ability to capture quantitative temporal structure. Second, although moderate context can improve LLM accuracy, adding further user-level detail degrades performance. These results challenge the assumption that "more context leads to better reasoning". Our study highlights fundamental limitations of today's LLMs in structured temporal inference and offers guidance for designing future context-aware hybrid models that integrate statistical precision with linguistic flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.07314v2">MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve superhuman performance on standardized medical licensing exams, these static benchmarks have become saturated and increasingly disconnected from the functional requirements of clinical workflows. To bridge the gap between theoretical capability and verified utility, we introduce MEDIC, a comprehensive evaluation framework establishing leading indicators across various clinical dimensions. Beyond standard question-answering, we assess operational capabilities using deterministic execution protocols and a novel Cross-Examination Framework (CEF), which quantifies information fidelity and hallucination rates without reliance on reference texts. Our evaluation across a heterogeneous task suite exposes critical performance trade-offs: we identify a significant knowledge-execution gap, where proficiency in static retrieval does not predict success in operational tasks such as clinical calculation or SQL generation. Furthermore, we observe a divergence between passive safety (refusal) and active safety (error detection), revealing that models fine-tuned for high refusal rates often fail to reliably audit clinical documentation for factual accuracy. These findings demonstrate that no single architecture dominates across all dimensions, highlighting the necessity of a portfolio approach to clinical model deployment. As part of this investigation, we released a public leaderboard on Hugging Face.\footnote{https://huggingface.co/spaces/m42-health/MEDIC-Benchmark}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12869v2">On the Fundamental Limits of LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Submitted to TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22922v4">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. Chain-of-thought (CoT) reasoning capabilities [1, 2] are well-known to help LLMs decompose a problem into steps and explore the solution spaces more effectively, leading to impressive performance on mathematical and reasoning benchmarks. As the length of CoT tokens per question increases substantially to even thousands of tokens per question [ 1], it is unknown how users could comprehend LLM reasoning and detect errors or hallucinations. To address this problem and understand how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph). That is, we ask LLMs themselves to generate an interactive web interface wrapped around the original CoT content, which may be presented in text (iCoT), graphs (iGraph) or code (iPoT). This interface allows users to interact with and provide a novel experience in reading and validating the reasoning chains of LLMs. Across a study of 125 participants, interactive interfaces significantly improve user performance. Specifically, iGraph users score the highest error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also lead to faster user validation time-iGraph users are faster (57.9 secs per question) than the users of iCoT and iPoT (60 secs) and the standard CoT (64.7 secs). A post-study questionnaire shows that users prefer iGraph, citing its superior ability to enable them to follow the LLM's reasoning. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18150v1">FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18146v1">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to ranking tasks in retrieval and recommendation. Although reasoning prompting can enhance ranking utility, our preliminary exploration reveals that its benefits are inconsistent and come at a substantial computational cost, suggesting that when to reason is as crucial as how to reason. To address this issue, we propose a reasoning routing framework that employs a lightweight, plug-and-play router head to decide whether to use direct inference (Non-Think) or reasoning (Think) for each instance before generation. The router head relies solely on pre-generation signals: i) compact ranking-aware features (e.g., candidate dispersion) and ii) model-aware difficulty signals derived from a diagnostic checklist reflecting the model's estimated need for reasoning. By leveraging these features before generation, the router outputs a controllable token that determines whether to apply the Think mode. Furthermore, the router can adaptively select its operating policy along the validation Pareto frontier during deployment, enabling dynamic allocation of computational resources toward instances most likely to benefit from Think under varying system constraints. Experiments on three public ranking datasets with different scales of open-source LLMs show consistent improvements in ranking utility with reduced token consumption (e.g., +6.3\% NDCG@10 with -49.5\% tokens on MovieLens with Qwen3-4B), demonstrating reasoning routing as a practical solution to the accuracy-efficiency trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09053v2">Who Fails Where? LLM and Human Error Patterns in Endometriosis Ultrasound Report Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In this study, we evaluate a locally-deployed large-language model (LLM) to convert unstructured endometriosis transvaginal ultrasound (eTVUS) scan reports into structured data for imaging informatics workflows. Across 49 eTVUS reports, we compared three LLMs (7B/8B and a 20B-parameter model) against expert human extraction. The 20B model achieved a mean accuracy of 86.02%, substantially outperforming smaller models and confirming the importance of scale in handling complex clinical text. Crucially, we identified a highly complementary error profile: the LLM excelled at syntactic consistency (e.g., date/numeric formatting) where humans faltered, while human experts provided superior semantic and contextual interpretation. We also found that the LLM's semantic errors were fundamental limitations that could not be mitigated by simple prompt engineering. These findings strongly support a human-in-the-loop (HITL) workflow in which the on-premise LLM serves as a collaborative tool, not a full replacement. It automates routine structuring and flags potential human errors, enabling imaging specialists to focus on high-level semantic validation. We discuss implications for structured reporting and interactive AI systems in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13481v2">neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Envy shapes competitiveness and cooperation in human groups, yet its role in large language model interactions remains largely unexplored. As LLMs increasingly operate in multi-agent settings, it is important to examine whether they exhibit envy-like preferences under social comparison. We evaluate LLM behavior across two scenarios: (1) a point-allocation game testing sensitivity to relative versus absolute payoff, and (2) comparative evaluations across general and contextual settings. To ground our analysis in psychological theory, we adapt four established psychometric questionnaires spanning general, domain-specific, workplace, and sibling-based envy. Our results reveal heterogeneous envy-like patterns across models and contexts, with some models sacrificing personal gain to reduce a peer's advantage, while others prioritize individual maximization. These findings highlight competitive dispositions as a design and safety consideration for multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18119v1">Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      SQL is central to enterprise data engineering, yet generating fully correct SQL code in a single attempt remains difficult, even for experienced developers and advanced text-to-SQL LLMs, often requiring multiple debugging iterations. We introduce OurBench, the first benchmark for enterprise-level SQL reasoning and debugging. Our benchmark is built on two key innovations: (1) an automated construction workflow that uses reverse engineering to systematically inject realistic bugs into large-scale SQL code, enabling scalable and diverse benchmark generation; and (2) an execution-free evaluation framework tailored to enterprise settings, providing fast, accurate, and resource-efficient assessment. OurBench comprises 469 OurBenchSyn queries featuring syntax errors with explicit error messages, and 516 OurBenchSem queries targeting semantic errors in which the code fails to meet user intent. The queries are highly complex, averaging over 140 lines and featuring deep and wide abstract syntax trees. Evaluation of nearly 30 LLMs reveals a substantial performance gap: the best-performing model, Claude-4-Sonnet, achieves only 36.46 percent accuracy on OurBenchSyn and 32.17 percent on OurBenchSem, while most models score below 20 percent. We further explore four solution strategies, identify key challenges, and outline promising directions for enterprise SQL debugging with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18116v1">FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis. We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs. Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18110v1">AttenMIA: LLM Membership Inference Attack through Attention Signals</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed to enable or improve a multitude of real-world applications. Given the large size of their training data sets, their tendency to memorize training data raises serious privacy and intellectual property concerns. A key threat is the membership inference attack (MIA), which aims to determine whether a given sample was included in the model's training set. Existing MIAs for LLMs rely primarily on output confidence scores or embedding-based features, but these signals are often brittle, leading to limited attack success. We introduce AttenMIA, a new MIA framework that exploits self-attention patterns inside the transformer model to infer membership. Attention controls the information flow within the transformer, exposing different patterns for memorization that can be used to identify members of the dataset. Our method uses information from attention heads across layers and combines them with perturbation-based divergence metrics to train an effective MIA classifier. Using extensive experiments on open-source models including LLaMA-2, Pythia, and Opt models, we show that attention-based features consistently outperform baselines, particularly under the important low-false-positive metric (e.g., achieving up to 0.996 ROC AUC & 87.9% TPR@1%FPR on the WikiMIA-32 benchmark with Llama2-13b). We show that attention signals generalize across datasets and architectures, and provide a layer- and head-level analysis of where membership leakage is most pronounced. We also show that using AttenMIA to replace other membership inference attacks in a data extraction framework results in training data extraction attacks that outperform the state of the art. Our findings reveal that attention mechanisms, originally introduced to enhance interpretability, can inadvertently amplify privacy risks in LLMs, underscoring the need for new defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18096v1">Enhancing LLM-based Recommendation with Preference Hint Discovery from Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs have garnered substantial attention in recommendation systems. Yet they fall short of traditional recommenders when capturing complex preference patterns. Recent works have tried integrating traditional recommendation embeddings into LLMs to resolve this issue, yet a core gap persists between their continuous embedding and discrete semantic spaces. Intuitively, textual attributes derived from interactions can serve as critical preference rationales for LLMs' recommendation logic. However, directly inputting such attribute knowledge presents two core challenges: (1) Deficiency of sparse interactions in reflecting preference hints for unseen items; (2) Substantial noise introduction from treating all attributes as hints. To this end, we propose a preference hint discovery model based on the interaction-integrated knowledge graph, enhancing LLM-based recommendation. It utilizes traditional recommendation principles to selectively extract crucial attributes as hints. Specifically, we design a collaborative preference hint extraction schema, which utilizes semantic knowledge from similar users' explicit interactions as hints for unseen items. Furthermore, we develop an instance-wise dual-attention mechanism to quantify the preference credibility of candidate attributes, identifying hints specific to each unseen item. Using these item- and user-based hints, we adopt a flattened hint organization method to shorten input length and feed the textual hint information to the LLM for commonsense reasoning. Extensive experiments on both pair-wise and list-wise recommendation tasks verify the effectiveness of our proposed framework, indicating an average relative improvement of over 3.02% against baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18091v1">From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly costly to deploy, motivating extensive research on model pruning. However, most existing studies focus on instruction-following LLMs, leaving it unclear whether established pruning strategies transfer to reasoning-augmented models that explicitly generate long intermediate reasoning traces. In this work, we conduct a controlled study of pruning for both instruction-following ($\textbf{LLM-instruct}$) and reasoning-augmented ($\textbf{LLM-think}$) models. To isolate the effects of pruning, we align pruning calibration and post-pruning recovery data with each model's original training distribution, which we show yields more stable and reliable pruning behavior. We evaluate static depth pruning, static width pruning, and dynamic pruning across 17 tasks spanning classification, generation, and reasoning. Our results reveal clear paradigm-dependent differences: depth pruning outperforms width pruning on classification tasks, while width pruning is more robust for generation and reasoning. Moreover, static pruning better preserves reasoning performance, whereas dynamic pruning excels on classification and generation but remains challenging for long-chain reasoning. These findings underscore the need for pruning strategies that explicitly account for the distinct characteristics of reasoning-augmented LLMs. Our code is publicly available at https://github.com/EIT-NLP/LRM-Pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16432v2">iPDB -- Optimizing SQL Queries with ML and LLM Predicates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Structured Query Language (SQL) has remained the standard query language for databases. SQL is highly optimized for processing structured data laid out in relations. Meanwhile, in the present application development landscape, it is highly desirable to utilize the power of learned models to perform complex tasks. Large language models (LLMs) have been shown to understand and extract information from unstructured textual data. However, SQL as a query language and accompanying relational database systems are either incompatible or inefficient for workloads that require leveraging learned models. This results in complex engineering and multiple data migration operations that move data between the data sources and the model inference platform. In this paper, we present iPDB, a relational system that supports in-database machine learning (ML) and large language model (LLM) inferencing using extended SQL syntax. In iPDB, LLMs and ML calls can function as semantic projects, as predicates to perform semantic selects and semantic joins, or for semantic grouping in group-by clauses. iPDB has a novel relational predict operator and semantic query optimizations that enable users to write and efficiently execute semantic SQL queries, outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13358v5">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 We resolved some issues in this paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18077v1">Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18067v1">EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 17 pages, 6 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Verilog's design cycle is inherently labor-intensive and necessitates extensive domain expertise. Although Large Language Models (LLMs) offer a promising pathway toward automation, their limited training data and intrinsic sequential reasoning fail to capture the strict formal logic and concurrency inherent in hardware systems. To overcome these barriers, we present EvolVE, the first framework to analyze multiple evolution strategies on chip design tasks, revealing that Monte Carlo Tree Search (MCTS) excels at maximizing functional correctness, while Idea-Guided Refinement (IGR) proves superior for optimization. We further leverage Structured Testbench Generation (STG) to accelerate the evolutionary process. To address the lack of complex optimization benchmarks, we introduce IC-RTL, targeting industry-scale problems derived from the National Integrated Circuit Contest. Evaluations establish EvolVE as the new state-of-the-art, achieving 98.1% on VerilogEval v2 and 92% on RTLLM v2. Furthermore, on the industry-scale IC-RTL suite, our framework surpasses reference implementations authored by contest participants, reducing the Power, Performance, Area (PPA) product by up to 66% in Huffman Coding and 17% in the geometric mean across all problems. The source code of the IC-RTL benchmark is available at https://github.com/weiber2002/ICRTL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18053v1">Addressing LLM Diversity by Infusing Random Concepts</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to produce outputs with limited diversity. In this work, we study whether infusing random concepts in the prompts can improve the diversity of the generated outputs. To benchmark the approach, we design a systematic evaluation protocol which involves prompting an LLM with questions of the form "Name 10 Hollywood actors", and analyzing diversity measures of the resulting LLM outputs. Our experiments on multiple LLMs show that prepending random words/sentences unrelated to the prompt result in greater diversity in the outputs of LLMs. We believe that this promising result and the evaluation protocol opens up interesting avenues for future work, such as how infusing randomness into LLMs could be applied to other domains. Further, the evaluation protocol could also inspire research into benchmarking LLM diversity more systematically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16891v4">LLMs as Layout Designers: Enhanced Spatial Reasoning for Content-Aware Layout Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their ability to understand and manipulate spatial relationships remains limited. Such capabilities are crucial for content-aware graphic layout design, where the goal is to arrange heterogeneous elements onto a canvas so that final design remains visually balanced and structurally feasible. This problem requires precise coordination of placement, alignment, and structural organization of multiple elements within a constrained visual space. To address this limitation, we introduce LaySPA, a reinforcement learning-based framework that augments LLM-based agents with explicit spatial reasoning capabilities for layout design. LaySPA employs hybrid reward signals that jointly capture geometric constraints, structural fidelity, and visual quality, enabling agents to navigate the canvas, model inter-element relationships, and optimize spatial arrangements. Through group-relative policy optimization, the agent generates content-aware layouts that reflect salient regions, respect spatial constraints, and produces an interpretable reasoning trace explaining placement decisions and a structured layout specification. Experimental results show that LaySPA substantially improves the generation of structurally valid and visually appealing layouts, outperforming larger general-purpose LLMs and achieving performance comparable to state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16034v2">Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18009v1">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at the 48th European Conference on Information Retrieval (ECIR 2026)
    </div>
    <details class="paper-abstract">
      Implicit feedback -- the main data source for training Recommender Systems (RSs) -- is inherently noisy and has been shown to negatively affect recommendation effectiveness. Denoising has been proposed as a method for removing noisy implicit feedback and improving recommendations. Prior work has focused on in-training denoising, however this requires additional data, changes to the model architecture and training procedure or fine-tuning, all of which can be costly and data hungry. In this work, we focus on post-training denoising. Different from in-training denoising, post-training denoising does not involve changing the architecture of the model nor its training procedure, and does not require additional data. Specifically, we present a method for post-training denoising user profiles using Large Language Models (LLMs) for Collaborative Filtering (CF) recommendations. Our approach prompts LLMs with (i) a user profile (user interactions), (ii) a candidate item, and (iii) its rank as given by the CF recommender, and asks the LLM to remove items from the user profile to improve the rank of the candidate item. Experiments with a state-of-the-art CF recommender and 4 open and closed source LLMs in 3 datasets show that our denoising yields improvements up to 13% in effectiveness over the original user profiles. Our code is available at https://github.com/edervishaj/denoising-user-profiles-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09897v2">PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 WWW 2026
    </div>
    <details class="paper-abstract">
      Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15979v2">OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17971v1">LLMs as Cultural Archives: Cultural Commonsense Knowledge Graph Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 EACL 2026 MAIN
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode rich cultural knowledge learned from diverse web-scale data, offering an unprecedented opportunity to model cultural commonsense at scale. Yet this knowledge remains mostly implicit and unstructured, limiting its interpretability and use. We present an iterative, prompt-based framework for constructing a Cultural Commonsense Knowledge Graph (CCKG) that treats LLMs as cultural archives, systematically eliciting culture-specific entities, relations, and practices and composing them into multi-step inferential chains across languages. We evaluate CCKG on five countries with human judgments of cultural relevance, correctness, and path coherence. We find that the cultural knowledge graphs are better realized in English, even when the target culture is non-English (e.g., Chinese, Indonesian, Arabic), indicating uneven cultural encoding in current LLMs. Augmenting smaller LLMs with CCKG improves performance on cultural reasoning and story generation, with the largest gains from English chains. Our results show both the promise and limits of LLMs as cultural technologies and that chain-structured cultural knowledge is a practical substrate for culturally grounded NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v3">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17942v1">LLM-Based SQL Generation: Prompting, Self-Refinement, and Adaptive Weighted Majority Voting</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL has emerged as a prominent research area, particularly with the rapid advancement of large language models (LLMs). By enabling users to query databases through natural language rather than SQL, this technology significantly lowers the barrier to data analysis. However, generating accurate SQL from natural language remains challenging due to ambiguity in user queries, the complexity of schema linking, limited generalization across SQL dialects, and the need for domain-specific understanding. In this study, we propose a Single-Agent Self-Refinement with Ensemble Voting (SSEV) pipeline built on PET-SQL that operates without ground-truth data, integrating self-refinement with Weighted Majority Voting (WMV) and its randomized variant (RWMA). Experimental results show that the SSEV achieves competitive performance across multiple benchmarks, attaining execution accuracies of 85.5% on Spider 1.0-Dev, 86.4% on Spider 1.0-Test, and 66.3% on BIRD-Dev. Building on insights from the SSEV pipeline, we further propose ReCAPAgent-SQL (Refinement-Critique-Act-Plan agent-based SQL framework) to address the growing complexity of enterprise databases and real-world Text-to-SQL tasks. The framework integrates multiple specialized agents for planning, external knowledge retrieval, critique, action generation, self-refinement, schema linking, and result validation, enabling iterative refinement of SQL predictions through agent collaboration. ReCAPAgent-SQL's WMA results achieve 31% execution accuracy on the first 100 queries of Spider 2.0-Lite, demonstrating significant improvements in handling real-world enterprise scenarios. Overall, our work facilitates the deployment of scalable Text-to-SQL systems in practical settings, supporting better data-driven decision-making at lower cost and with greater efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10769v3">Mind the Gap: Benchmarking LLM Uncertainty and Calibration with Specialty-Aware Clinical QA and Reasoning-Based Behavioural Features</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at EACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential when employing large language models (LLMs) in high-risk domains such as clinical question answering (QA). In this work, we evaluate uncertainty estimation methods for clinical QA focusing, for the first time, on eleven clinical specialties and six question types, and across ten open-source LLMs (general-purpose, biomedical, and reasoning models), alongside representative proprietary models. We analyze score-based UQ methods, present a case study introducing a novel lightweight method based on behavioral features derived from reasoning-oriented models, and examine conformal prediction as a complementary set-based approach. Our findings reveal that uncertainty reliability is not a monolithic property, but one that depends on clinical specialty and question type due to shifts in calibration and discrimination. Our results highlight the need to select or ensemble models based on their distinct, complementary strengths and clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17915v1">Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context. We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09635v2">LLM for Large-Scale Optimization Model Auto-Formulation: Bridging Flexibility and Standardization via Agentic Workflow</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Updated version of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027
    </div>
    <details class="paper-abstract">
      Large-scale optimization is a key backbone of modern business decision-making. However, building these models is often labor-intensive and time-consuming. We address this by proposing LEAN-LLM-OPT, a LightwEight AgeNtic workflow construction framework for LLM-assisted large-scale OPTimization auto-formulation. LEAN-LLM-OPT takes as input a problem description together with associated datasets and orchestrates a team of LLM agents to produce an optimization formulation. Specifically, upon receiving a query, two upstream LLM agents dynamically construct a workflow that specifies, step-by-step, how optimization models for similar problems can be formulated. A downstream LLM agent then follows this workflow to generate the final output. The agentic workflow leverages common modeling practices to standardize the modeling process into a sequence of structured sub-tasks, offloading mechanical data-handling operations to auxiliary tools. This reduces the LLM's burden in planning and data handling, allowing us to exploit its flexibility to address unstructured components. Extensive simulations show that LEAN-LLM-OPT, instantiated with GPT-4.1 and the open source gpt-oss-20B, achieves strong performance on large-scale optimization modeling tasks and is competitive with state-of-the-art approaches. In addition, in a Singapore Airlines choice-based revenue management use case, LEAN-LLM-OPT demonstrates practical value by achieving leading performance across a range of scenarios. Along the way, we introduce Large-Scale-OR and Air-NRM, the first comprehensive benchmarks for large-scale optimization auto-formulation. The code and data of this work is available at https://github.com/CoraLiang01/lean-llm-opt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17902v1">dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Automatic speech recognition (ASR) systems based on large language models (LLMs) achieve superior performance by leveraging pretrained LLMs as decoders, but their token-by-token generation mechanism leads to inference latency that grows linearly with sequence length. Meanwhile, discrete diffusion large language models (dLLMs) offer a promising alternative, enabling high-quality parallel sequence generation with pretrained decoders. However, directly applying native text-oriented dLLMs to ASR leads to a fundamental mismatch between open-ended text generation and the acoustically conditioned transcription paradigm required by ASR. As a result, it introduces unnecessary difficulty and computational redundancy, such as denoising from pure noise, inflexible generation lengths, and fixed denoising steps. We propose dLLM-ASR, an efficient dLLM-based ASR framework that formulates dLLM's decoding as a prior-guided and adaptive denoising process. It leverages an ASR prior to initialize the denoising process and provide an anchor for sequence length. Building upon this prior, length-adaptive pruning dynamically removes redundant tokens, while confidence-based denoising allows converged tokens to exit the denoising loop early, enabling token-level adaptive computation. Experiments demonstrate that dLLM-ASR achieves recognition accuracy comparable to autoregressive LLM-based ASR systems and delivers a 4.44$\times$ inference speedup, establishing a practical and efficient paradigm for ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19113v3">Argument-Based Consistency in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 7 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity - from their explanations that justify a stance - to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Argument-based Consistency (ArC), that measures the extent to which LLMs' free-form toxicity explanations reflect an ideal and logical argumentation process. Based on uncertainty quantification, we develop six metrics for ArC to comprehensively evaluate the (in)consistencies in LLMs' toxicity explanations. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code (https://github.com/uofthcdslab/ArC) and LLM-generated explanations (https://huggingface.co/collections/uofthcdslab/arc) for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17897v1">UniCog: Uncovering Cognitive Abilities of LLMs through Latent Mind Space Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A growing body of research suggests that the cognitive processes of large language models (LLMs) differ fundamentally from those of humans. However, existing interpretability methods remain limited in explaining how cognitive abilities are engaged during LLM reasoning. In this paper, we propose UniCog, a unified framework that analyzes LLM cognition via a latent mind space. Formulated as a latent variable model, UniCog encodes diverse abilities from dense model activations into sparse, disentangled latent dimensions. Through extensive analysis on six advanced LLMs, including DeepSeek-V3.2 and GPT-4o, we reveal a Pareto principle of LLM cognition, where a shared reasoning core is complemented by ability-specific signatures. Furthermore, we discover that reasoning failures often manifest as anomalous intensity in latent activations. These findings opens a new paradigm in LLM analysis, providing a cognition grounded view of reasoning dynamics. Finally, leveraging these insights, we introduce a latent-informed candidate prioritization strategy, which improves reasoning performance by up to 7.5% across challenging benchmarks. Our code is available at https://github.com/milksalute/unicog.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v3">TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures reasoning models not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly released TruthTensor at https://truthtensor.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10978v3">Does LLM Focus on the Right Words? Mitigating Context Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Context Bias, whereby the model over-relies on auxiliary tokens, such as task descriptions and prefix-generated tokens, while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating context bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates context bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. The code is available at https://github.com/WANGBohaO-jpg/GDRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14506v4">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real robot platforms, including a Toyota Human Support Robot and an ANYmal D robot with a Unitree Z1 arm. Our method achieves a 95% success rate in the `fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art LLM-based robotics planners, while using only real-time onboard computing. Project website: https://kimtienly.github.io/InteLiPlan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17817v1">Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT: An LLM-Enhanced Agentic AI Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      The rapid expansion of low-altitude economy Internet of Things (LAE-IoT) networks has created unprecedented security challenges due to dynamic three-dimensional mobility patterns, distributed autonomous operations, and severe resource constraints. Traditional intrusion detection systems designed for static ground-based networks prove inadequate for tackling the unique characteristics of aerial IoT environments, including frequent topology changes, real-time detection requirements, and energy limitations. In this article, we analyze the intrusion detection requirements for LAE-IoT networks, complemented by a comprehensive review of evaluation metrics that cover detection effectiveness, response time, and resource consumption. Then, we investigate transformative potential of agentic artificial intelligence (AI) paradigms and introduce a large language model (LLM)-enabled agentic AI framework for enhancing intrusion detection in LAE-IoT networks. This leads to our proposal of a novel multi-agent collaborative intrusion detection framework that leverages specialized LLM-enhanced agents for intelligent data processing and adaptive classification. Through experimental validation, our framework demonstrates superior performance of over 90\% classification accuracy across multiple benchmark datasets. These results highlight the transformative potential of combining agentic AI principles with LLMs for next-generation LAE-IoT security systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17814v1">MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have advanced rapidly, yet heterogeneity in architecture, alignment strategies, and efficiency means that no single model is uniformly superior across tasks. In practical deployments, workloads span lightweight OCR to complex multimodal reasoning; using one MLLM for all queries either over-provisions compute on easy instances or sacrifices accuracy on hard ones. Query-level model selection (routing) addresses this tension, but extending routing from text-only LLMs to MLLMs is nontrivial due to modality fusion, wide variation in computational cost across models, and the absence of a standardized, budget-aware evaluation. We present MMR-Bench, a unified benchmark that isolates the multimodal routing problem and enables comparison under fixed candidate sets and cost models. MMR-Bench provides (i) a controlled environment with modality-aware inputs and variable compute budgets, (ii) a broad suite of vision-language tasks covering OCR, general VQA, and multimodal math reasoning, and (iii) strong single-model reference, oracle upper bounds, and representative routing policies. Using MMR-Bench, we show that incorporating multimodal signals improves routing quality. Empirically, these cues improve the cost-accuracy frontier and enable the routed system to exceed the strongest single model's accuracy at roughly 33% of its cost. Furthermore, policies trained on a subset of models and tasks generalize zero-shot to new datasets and text-only benchmarks without retuning, establishing MMR-Bench as a foundation for studying adaptive multimodal model selection and efficient MLLM deployment. The code will be available at: https://github.com/Hunter-Wrynn/MMR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15333v2">Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess strong representation and reasoning capabilities, but their application to structure-based drug design (SBDD) is limited by insufficient understanding of protein structures and unpredictable molecular generation. To address these challenges, we propose Exploration-Augmented Latent Inference for LLMs (ELILLM), a framework that reinterprets the LLM generation process as an encoding, latent space exploration, and decoding workflow. ELILLM explicitly explores portions of the design problem beyond the model's current knowledge while using a decoding module to handle familiar regions, generating chemically valid and synthetically reasonable molecules. In our implementation, Bayesian optimization guides the systematic exploration of latent embeddings, and a position-aware surrogate model efficiently predicts binding affinity distributions to inform the search. Knowledge-guided decoding further reduces randomness and effectively imposes chemical validity constraints. We demonstrate ELILLM on the CrossDocked2020 benchmark, showing strong controlled exploration and high binding affinity scores compared with seven baseline methods. These results demonstrate that ELILLM can effectively enhance LLMs capabilities for SBDD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17789v1">Neuro-Symbolic Verification on Instruction Following of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A fundamental problem of applying Large Language Models (LLMs) to important applications is that LLMs do not always follow instructions, and violations are often hard to observe or check. In LLM-based agentic workflows, such violations can propagate and amplify along reasoning chains, causing task failures and system incidents. This paper presents NSVIF, a neuro-symbolic framework for verifying whether an LLM's output follows the instructions used to prompt the LLM. NSVIF is a universal, general-purpose verifier; it makes no assumption about the instruction or the LLM. NSVIF formulates instruction-following verification as a constraint-satisfaction problem by modeling user instructions as constraints. NSVIF models both logical and semantic constraints; constraint solving is done by a unified solver that orchestrates logical reasoning and semantic analysis. To evaluate NSVIF, we develop VIFBENCH, a new benchmark for instruction-following verifiers with fine-grained data labels. Experiments show that NSVIF significantly outperforms LLM-based approaches and provides interpretable feedback. We also show that feedback from NSVIF helps improve LLMs' instruction-following capability without post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15331v2">DesignerlyLoop: Forming Design Intent through Curated Reasoning for Human-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) show promise in design tasks, yet a fundamental misalignment persists: design thinking requires iterative intent formulation, while LLMs treat inputs as complete specifications. This challenges design intent formulation, where designers must progressively refine understanding through exploration. Existing tools either sacrifice exploratory flexibility for structural stability or leave reasoning implicit, failing to support human-LLM alignment. Through a formative study with eight designers, we introduce curated reasoning-enabling designers to explicitly inspect, reorganize, and selectively regenerate LLM reasoning structures. We present DesignerlyLoop, implementing this through a two-layer structure separating design intent from LLM reasoning. A study with 20 designers demonstrates that curated reasoning significantly improves design quality and creativity. Our work contributes a novel interaction paradigm for human-LLM alignment, transforming LLMs from content generators into structured reasoning partners in creative design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17769v1">Reflexa: Uncovering How LLM-Supported Reflection Scaffolding Reshapes Creativity in Creative Coding</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Creative coding requires continuous translation between evolving concepts and computational artifacts, making reflection essential yet difficult to sustain. Creators often struggle to manage ambiguous intentions, emergent outputs, and complex code, limiting depth of exploration. This work examines how large language models (LLMs) can scaffold reflection not as isolated prompts, but as a system-level mechanism shaping creative regulation. From formative studies with eight expert creators, we derived reflection challenges and design principles that informed Reflexa, an integrated scaffold combining dialogic guidance, visualized version navigation, and iterative suggestion pathways. A within-subject study with 18 participants provides an exploratory mechanism validation, showing that structured reflection patterns mediate the link between AI interaction and creative outcomes. These reflection trajectories enhanced perceived controllability, broadened exploration, and improved originality and aesthetic quality. Our findings advance HCI understanding of reflection from LLM-assisted creative practices, and provide design strategies for building LLM-based creative tools that support richer human-AI co-creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17768v1">LLM-42: Enabling Determinism in LLM Inference with Verified Speculation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 https://github.com/microsoft/llm-42
    </div>
    <details class="paper-abstract">
      In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. A straightforward way to eliminate non-determinism is to disable dynamic batching during inference, but doing so severely degrades throughput. Another approach is to make kernels batch-invariant; however, this tightly couples determinism to kernel design, requiring new implementations. This coupling also imposes fixed runtime overheads, regardless of how much of the workload actually requires determinism. Inspired by ideas from speculative decoding, we present LLM-42, a scheduling-based approach to enable determinism in LLM inference. Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching. Moreover, most GPU kernels use shape-consistent reductions. Leveraging these insights, LLM-42 decodes tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback loop. The verifier replays candidate tokens under a fixed-shape reduction schedule, commits those that are guaranteed to be consistent across runs, and rolls back those violating determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to the traffic that requires determinism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v1">The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17716v1">Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Presented at the NeusymBridge Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many tasks but still struggle with a critical ability for LLM-based agents: asking good questions for resolving ambiguity in user requests. While prior work has explored information-seeking behavior through word games, existing benchmarks lack comprehensive evaluation frameworks that provide both final and intermediate signals based on Information Gain (IG). Moreover, they rarely provide systematic comparisons between models that use chain-of-thought reasoning and those that do not. We propose a multi-turn dialogue framework that quantitatively measures how effectively LLMs gather information through yes/no questions in a hierarchical knowledge graph environment. Our framework employs a triad of interacting LLM agents that ask questions, answer them, and update the hypothesis space. We adopt IG as the main metric, grounded in Shannon entropy, to assess query effectiveness at each turn and cumulatively. We instantiate our framework in a geographical Guess My City game setting organized in a five-level taxonomy and evaluate multiple LLM variants under fully and partially observable conditions, with and without Chain-of-Thought reasoning. Our experiments demonstrate that, among the evaluated models, the ones with explicit reasoning capabilities achieve higher IG per turn and reach solutions in fewer steps, particularly in partially observable settings. Analysis of reasoning traces reveals that smaller models compensate for limited capacity through more aggressive exploration of candidate questions, while larger models exhibit higher assertiveness in selecting optimal queries, generating candidates with greater potential IG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01659v3">From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17705v1">Distance-to-Distance Ratio: A Similarity Measure for Sentences Based on Rate of Change in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      A measure of similarity between text embeddings can be considered adequate only if it adheres to the human perception of similarity between texts. In this paper, we introduce the distance-to-distance ratio (DDR), a novel measure of similarity between LLM sentence embeddings. Inspired by Lipschitz continuity, DDR measures the rate of change in similarity between the pre-context word embeddings and the similarity between post-context LLM embeddings, thus measuring the semantic influence of context. We evaluate the performance of DDR in experiments designed as a series of perturbations applied to sentences drawn from a sentence dataset. For each sentence, we generate variants by replacing one, two, or three words with either synonyms, which constitute semantically similar text, or randomly chosen words, which constitute semantically dissimilar text. We compare the performance of DDR with other prevailing similarity metrics and demonstrate that DDR consistently provides finer discrimination between semantically similar and dissimilar texts, even under minimal, controlled edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17692v1">LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 31pages, 4 figures
    </div>
    <details class="paper-abstract">
      Statute retrieval is essential for legal assistance and judicial decision support, yet real-world legal queries are often implicit, multi-issue, and expressed in colloquial or underspecified forms. These characteristics make it difficult for conventional retrieval-augmented generation pipelines to recover the statutory elements required for accurate retrieval. Dense retrievers focus primarily on the literal surface form of the query, whereas lightweight rerankers lack the legal-reasoning capacity needed to assess statutory applicability. We present LegalMALR, a retrieval framework that integrates a Multi-Agent Query Understanding System (MAS) with a zero-shot large-language-model-based reranking module (LLM Reranker). MAS generates diverse, legally grounded reformulations and conducts iterative dense retrieval to broaden candidate coverage. To stabilise the stochastic behaviour of LLM-generated rewrites, we optimise a unified MAS policy using Generalized Reinforcement Policy Optimization(GRPO). The accumulated candidate set is subsequently evaluated by the LLM Reranker, which performs natural-language legal reasoning to produce the final ranking. We further construct CSAID, a dataset of 118 difficult Chinese legal queries annotated with multiple statutory labels, and evaluate LegalMALR on both CSAID and the public STARD benchmark. Experiments show that LegalMALR substantially outperforms strong Retrieval-augmented generation(RAG) baselines in both in-distribution and out-of-distribution settings, demonstrating the effectiveness of combining multi-perspective query interpretation, reinforcement-based policy optimisation, and large-model reranking for statute retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17676v1">GazeSummary: Exploring Gaze as an Implicit Prompt for Personalization in Text-based LLM Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Smart glasses are accelerating progress toward more seamless and personalized LLM-based assistance by integrating multimodal inputs. Yet, these inputs rely on obtrusive explicit prompts. The advent of gaze tracking on smart devices offers a unique opportunity to extract implicit user intent for personalization. This paper investigates whether LLMs can interpret user gaze for text-based tasks. We evaluate different gaze representations for personalization and validate their effectiveness in realistic reading tasks. Results show that LLMs can leverage gaze to generate high-quality personalized summaries and support users in downstream tasks, highlighting the feasibility and value of gaze-driven personalization for future mobile and wearable LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00003v2">A Review of Incorporating Psychological Theories in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Psychological insights have long shaped pivotal NLP breakthroughs, from attention mechanisms to reinforcement learning and social modeling. As Large Language Models (LLMs) develop, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development. Our review integrates insights from six subfields of psychology, including cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. With stage-wise analysis, we highlight current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04537v2">Not All Steps are Informative: On the Linearity of LLMs' RLVR Training</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on mathematics and code benchmarks by extrapolating beyond the step range where RL training remains stable. Our code is available at https://github.com/Miaow-Lab/RLVR-Linearity
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17668v1">Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. The results are consistent across a wide range of tasks, including long-context understanding, code comprehension, and mathematical reasoning, demonstrating the generality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v3">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02547v4">The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Published on Transactions on Machine Learning Research: https://openreview.net/forum?id=RY19y2RI1O
    </div>
    <details class="paper-abstract">
      The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17614v1">AlignUI: A Method for Designing LLM-Generated UIs Aligned with User Preferences</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Designing user interfaces that align with user preferences is a time-consuming process, which requires iterative cycles of prototyping, user testing, and refinement. Recent advancements in LLM-based UI generation have enabled efficient UI generation to assist the UI design process. We introduce AlignUI, a method that aligns LLM-generated UIs with user tasks and preferences by using a user preference dataset to guide the LLM's reasoning process. The dataset was crowdsourced from 50 general users (the target users of generated UIs) and contained 720 UI control preferences on eight image-editing tasks. We evaluated AlignUI by generating UIs for six unseen tasks and conducting a user study with 72 additional general users. The results showed that the generated UIs closely align with multiple dimensions of user preferences. We conclude by discussing the applicability of our method to support user-aligned UI design for multiple task domains and user groups, as well as personalized user needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17604v1">Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used to support software developers in tasks such as code generation, optimization, and documentation. However, their ability to improve existing programming answers in a human-like manner remains underexplored. On technical question-and-answer platforms such as Stack Overflow (SO), contributors often revise answers based on user comments that identify errors, inefficiencies, or missing explanations. Yet roughly one-third of this feedback is never addressed due to limited time, expertise, or visibility, leaving many answers incomplete or outdated. This study investigates whether LLMs can enhance programming answers by interpreting and incorporating comment-based feedback. We make four main contributions. First, we introduce ReSOlve, a benchmark consisting of 790 SO answers with associated comment threads, annotated for improvement-related and general feedback. Second, we evaluate four state-of-the-art LLMs on their ability to identify actionable concerns, finding that DeepSeek achieves the best balance between precision and recall. Third, we present AUTOCOMBAT, an LLM-powered tool that improves programming answers by jointly leveraging user comments and question context. Compared to human revised references, AUTOCOMBAT produces near-human quality improvements while preserving the original intent and significantly outperforming the baseline. Finally, a user study with 58 practitioners shows strong practical value, with 84.5 percent indicating they would adopt or recommend the tool. Overall, AUTOCOMBAT demonstrates the potential of scalable, feedback-driven answer refinement to improve the reliability and trustworthiness of technical knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21184v4">Jailbreak-as-a-Service++: Unveiling Distributed AI-Driven Malicious Information Campaigns Powered by LLM Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      To prevent the misuse of Large Language Models (LLMs) for malicious purposes, numerous efforts have been made to develop the safety alignment mechanisms of LLMs. However, as multiple LLMs become readily accessible through various Model-as-a-Service (MaaS) platforms, attackers can strategically exploit LLMs' heterogeneous safety policies to fulfill malicious information generation tasks in a distributed manner. In this study, we introduce \textit{\textbf{PoisonSwarm}} to how attackers can reliably launder malicious tasks via the speculative use of LLM crowdsourcing. Building upon a scheduler orchestrating crowdsourced LLMs, PoisonSwarm maps the given malicious task to a benign analogue to derive a content template, decomposes it into semantic units for crowdsourced unit-wise rewriting, and reassembles the outputs into malicious content. Experiments show its superiority over existing methods in data quality, diversity, and success rates. Regulation simulations further reveal the difficulty of governing such distributed, orchestrated misuse in MaaS ecosystems, highlighting the need for coordinated, ecosystem-level defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15955v2">How Good Are LLMs at Processing Tool Outputs?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Most realistic task automation problems require large language models (LLMs) to call tools, which often return complex JSON responses. These responses must be further processed to derive the information necessary for task completion. The ability of LLMs to do so is under-studied. In this paper, we study the tool response processing task and LLMs' abilities to process structured (JSON) responses. We created a dataset for this task, and evaluated 15 open and closed weight models using multiple prompting approaches. Our results show that JSON processing remains a difficult task even for frontier models across multiple prompting strategies. The optimal response processing strategy depends on both the nature and size of the tool outputs, as well as the complexity of the required reasoning. Variations in processing approaches can lead to performance differences ranging from 3\% to 50\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17593v1">From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Recent progress in large language models has renewed interest in mechanistically characterizing how multi-step reasoning is represented and computed. While much prior work treats reasoning as a linear chain of steps, many reasoning problems are more naturally structured as directed acyclic graphs (DAGs), where intermediate conclusions may depend on multiple premises, branch into parallel sub-derivations, and later merge or be reused. Understanding whether such graph-structured reasoning is reflected in model internals remains an open question. In this work, we introduce Reasoning DAG Probing, a framework that directly asks whether LLM hidden states encode the geometry of a reasoning DAG in a linearly accessible form, and where this structure emerges across layers. Within this framework, we associate each reasoning node with a textual realization and train lightweight probes to predict two graph-theoretic properties from hidden states: node depth and pairwise node distance. We use these probes to analyze the layerwise emergence of DAG structure and evaluate controls that disrupt reasoning-relevant structure while preserving superficial textual properties. Our results provide evidence that reasoning DAG geometry is meaningfully encoded in intermediate layers, with recoverability varying systematically by node depth and model scale, suggesting that LLM reasoning is not only sequential but exhibits measurable internal graph structure.
    </details>
</div>
