# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08243v1">Jinx: Unlimited LLMs for Probing Alignment Failures</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 https://huggingface.co/Jinx-org
    </div>
    <details class="paper-abstract">
      Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08242v1">Bringing Everyone to the Table: An Experimental Study of LLM-Facilitated Group Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Group decision-making often suffers from uneven information sharing, hindering decision quality. While large language models (LLMs) have been widely studied as aids for individuals, their potential to support groups of users, potentially as facilitators, is relatively underexplored. We present a pre-registered randomized experiment with 1,475 participants assigned to 281 five-person groups completing a hidden profile task--selecting an optimal city for a hypothetical sporting event--under one of four facilitation conditions: no facilitation, a one-time message prompting information sharing, a human facilitator, or an LLM (GPT-4o) facilitator. We find that LLM facilitation increases information shared within a discussion by raising the minimum level of engagement with the task among group members, and that these gains come at limited cost in terms of participants' attitudes towards the task, their group, or their facilitator. Whether by human or AI, there is no significant effect of facilitation on the final decision outcome, suggesting that even substantial but partial increases in information sharing are insufficient to overcome the hidden profile effect studied. To support further research into how LLM-based interfaces can support the future of collaborative decision making, we release our experimental platform, the Group-AI Interaction Laboratory (GRAIL), as an open-source tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08236v1">Exploring Safety Alignment Evaluation of LLMs in Chinese Mental Health Dialogues via LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Evaluating the safety alignment of LLM responses in high-risk mental health dialogues is particularly difficult due to missing gold-standard answers and the ethically sensitive nature of these interactions. To address this challenge, we propose PsyCrisis-Bench, a reference-free evaluation benchmark based on real-world Chinese mental health dialogues. It evaluates whether the model responses align with the safety principles defined by experts. Specifically designed for settings without standard references, our method adopts a prompt-based LLM-as-Judge approach that conducts in-context evaluation using expert-defined reasoning chains grounded in psychological intervention principles. We employ binary point-wise scoring across multiple safety dimensions to enhance the explainability and traceability of the evaluation. Additionally, we present a manually curated, high-quality Chinese-language dataset covering self-harm, suicidal ideation, and existential distress, derived from real-world online discourse. Experiments on 3600 judgments show that our method achieves the highest agreement with expert assessments and produces more interpretable evaluation rationales compared to existing approaches. Our dataset and evaluation tool are publicly available to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08221v1">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 26 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09373v2">QUDsim: Quantifying Discourse Similarities in LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 COLM 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      As large language models become increasingly capable at various writing tasks, their weakness at generating unique and creative content becomes a major liability. Although LLMs have the ability to generate text covering diverse topics, there is an overall sense of repetitiveness across texts that we aim to formalize and quantify via a similarity metric. The familiarity between documents arises from the persistence of underlying discourse structures. However, existing similarity metrics dependent on lexical overlap and syntactic patterns largely capture $\textit{content}$ overlap, thus making them unsuitable for detecting $\textit{structural}$ similarities. We introduce an abstraction based on linguistic theories in Questions Under Discussion (QUD) and question semantics to help quantify differences in discourse progression. We then use this framework to build $\textbf{QUDsim}$, a similarity metric that can detect discursive parallels between documents. Using QUDsim, we find that LLMs often reuse discourse structures (more so than humans) across samples, even when content differs. Furthermore, LLMs are not only repetitive and structurally uniform, but are also divergent from human authors in the types of structures they use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08211v1">SAEMark: Multi-bit LLM Watermarking with Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 24 pages, 12 figures, code available: https://zhuohaoyu.github.io/SAEMark
    </div>
    <details class="paper-abstract">
      Watermarking LLM-generated text is critical for content attribution and misinformation prevention. However, existing methods compromise text quality, require white-box model access and logit manipulation. These limitations exclude API-based models and multilingual scenarios. We propose SAEMark, a general framework for post-hoc multi-bit watermarking that embeds personalized messages solely via inference-time, feature-based rejection sampling without altering model logits or requiring training. Our approach operates on deterministic features extracted from generated text, selecting outputs whose feature statistics align with key-derived targets. This framework naturally generalizes across languages and domains while preserving text quality through sampling LLM outputs instead of modifying. We provide theoretical guarantees relating watermark success probability and compute budget that hold for any suitable feature extractor. Empirically, we demonstrate the framework's effectiveness using Sparse Autoencoders (SAEs), achieving superior detection accuracy and text quality. Experiments across 4 datasets show SAEMark's consistent performance, with 99.7% F1 on English and strong multi-bit detection accuracy. SAEMark establishes a new paradigm for scalable watermarking that works out-of-the-box with closed-source LLMs while enabling content attribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17130v3">Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector. Our code is publicly available at: https://github.com/hannahxchen/llm-censorship-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02904v2">How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. Our code is publicly available at https://github.com/HZD01/post-training-mechanistic-analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08171v1">PyVeritas: On Verifying Python via LLM-Based Transpilation and Bounded Model Checking for C</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 14 pages, 6 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Python has become the dominant language for general-purpose programming, yet it lacks robust tools for formal verification. In contrast, programmers working in languages such as C benefit from mature model checkers, for example CBMC, which enable exhaustive symbolic reasoning and fault localisation. The inherent complexity of Python, coupled with the verbosity and low-level nature of existing transpilers (e.g., Cython), have historically limited the applicability of formal verification to Python programs. In this paper, we propose PyVeritas, a novel framework that leverages Large Language Models (LLMs) for high-level transpilation from Python to C, followed by bounded model checking and MaxSAT-based fault localisation in the generated C code. PyVeritas enables verification and bug localisation for Python code using existing model checking tools for C. Our empirical evaluation on two Python benchmarks demonstrates that LLM-based transpilation can achieve a high degree of accuracy, up to 80--90% for some LLMs, enabling effective development environment that supports assertion-based verification and interpretable fault diagnosis for small yet non-trivial Python programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23701v2">TextQuests: How Good are LLMs at Text-Based Video Games?</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at https://textquests.ai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08147v1">From Natural Language to Solver-Ready Power System Optimization: An LLM-Assisted, Validation-in-the-Loop Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Large Language Models (LLMs)-assisted agent that automatically converts natural-language descriptions of power system optimization scenarios into compact, solver-ready formulations and generates corresponding solutions. In contrast to approaches that rely solely on LLM to produce solutions directly, the proposed method focuses on discovering a mathematically compatible formulation that can be efficiently solved by off-the-shelf optimization solvers. Directly using LLMs to produce solutions often leads to infeasible or suboptimal results, as these models lack the numerical precision and constraint-handling capabilities of established optimization solvers. The pipeline integrates a domain-aware prompt and schema with an LLM, enforces feasibility through systematic validation and iterative repair, and returns both solver-ready models and user-facing results. Using the unit commitment problem as a representative case study, the agent produces optimal or near-optimal schedules along with the associated objective costs. Results demonstrate that coupling the solver with task-specific validation significantly enhances solution reliability. This work shows that combining AI with established optimization frameworks bridges high-level problem descriptions and executable mathematical models, enabling more efficient decision-making in energy systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08139v1">Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08127v1">BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The security of LLM-based multi-agent systems (MAS) is critically threatened by propagation vulnerability, where malicious agents can distort collective decision-making through inter-agent message interactions. While existing supervised defense methods demonstrate promising performance, they may be impractical in real-world scenarios due to their heavy reliance on labeled malicious agents to train a supervised malicious detection model. To enable practical and generalizable MAS defenses, in this paper, we propose BlindGuard, an unsupervised defense method that learns without requiring any attack-specific labels or prior knowledge of malicious behaviors. To this end, we establish a hierarchical agent encoder to capture individual, neighborhood, and global interaction patterns of each agent, providing a comprehensive understanding for malicious agent detection. Meanwhile, we design a corruption-guided detector that consists of directional noise injection and contrastive learning, allowing effective detection model training solely on normal agent behaviors. Extensive experiments show that BlindGuard effectively detects diverse attack types (i.e., prompt injection, memory poisoning, and tool attack) across MAS with various communication patterns while maintaining superior generalizability compared to supervised baselines. The code is available at: https://github.com/MR9812/BlindGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08120v1">Vision-Based Localization and LLM-based Navigation for Indoor Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 20 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Indoor navigation remains a complex challenge due to the absence of reliable GPS signals and the architectural intricacies of large enclosed environments. This study presents an indoor localization and navigation approach that integrates vision-based localization with large language model (LLM)-based navigation. The localization system utilizes a ResNet-50 convolutional neural network fine-tuned through a two-stage process to identify the user's position using smartphone camera input. To complement localization, the navigation module employs an LLM, guided by a carefully crafted system prompt, to interpret preprocessed floor plan images and generate step-by-step directions. Experimental evaluation was conducted in a realistic office corridor with repetitive features and limited visibility to test localization robustness. The model achieved high confidence and an accuracy of 96% across all tested waypoints, even under constrained viewing conditions and short-duration queries. Navigation tests using ChatGPT on real building floor maps yielded an average instruction accuracy of 75%, with observed limitations in zero-shot reasoning and inference time. This research demonstrates the potential for scalable, infrastructure-free indoor navigation using off-the-shelf cameras and publicly available floor plans, particularly in resource-constrained settings like hospitals, airports, and educational institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08115v1">TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 10 pages, 1 figure, 6 tables(2 in main, 4 in appendix)
    </div>
    <details class="paper-abstract">
      We present TeamMedAgents, a novel multi-agent approach that systematically integrates evidence-based teamwork components from human-human collaboration into medical decision-making with large language models (LLMs). Our approach validates an organizational psychology teamwork model from human collaboration to computational multi-agent medical systems by operationalizing six core teamwork components derived from Salas et al.'s "Big Five" model: team leadership, mutual performance monitoring, team orientation, shared mental models, closed-loop communication, and mutual trust. We implement and evaluate these components as modular, configurable mechanisms within an adaptive collaboration architecture while assessing the effect of the number of agents involved based on the task's requirements and domain. Systematic evaluation of computational implementations of teamwork behaviors across eight medical benchmarks (MedQA, MedMCQA, MMLU-Pro Medical, PubMedQA, DDXPlus, MedBullets, Path-VQA, and PMC-VQA) demonstrates consistent improvements across 7 out of 8 evaluated datasets. Controlled ablation studies conducted on 50 questions per configuration across 3 independent runs provide mechanistic insights into individual component contributions, revealing optimal teamwork configurations that vary by reasoning task complexity and domain-specific requirements. Our ablation analyses reveal dataset-specific optimal teamwork configurations, indicating that different medical reasoning modalities benefit from distinct collaborative patterns. TeamMedAgents represents an advancement in collaborative AI by providing a systematic translation of established teamwork theories from human collaboration into agentic collaboration, establishing a foundation for evidence-based multi-agent system design in critical decision-making domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08096v1">Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Preprint as provided by the authors (19 pages, 12 figures, 9 tables)
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and their increased accessibility have made it easier than ever for students to automatically generate texts, posing new challenges for educational institutions. To enforce norms of academic integrity and ensure students' learning, learning analytics methods to automatically detect LLM-generated text appear increasingly appealing. This paper benchmarks the performance of different state-of-the-art detectors in educational contexts, introducing a novel dataset, called Generative Essay Detection in Education (GEDE), containing over 900 student-written essays and over 12,500 LLM-generated essays from various domains. To capture the diversity of LLM usage practices in generating text, we propose the concept of contribution levels, representing students' contribution to a given assignment. These levels range from purely human-written texts, to slightly LLM-improved versions, to fully LLM-generated texts, and finally to active attacks on the detector by "humanizing" generated texts. We show that most detectors struggle to accurately classify texts of intermediate student contribution levels, like LLM-improved human-written texts. Detectors are particularly likely to produce false positives, which is problematic in educational settings where false suspicions can severely impact students' lives. Our dataset, code, and additional supplementary materials are publicly available at https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08029v1">Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08027v1">Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Speech Recognition (ASR) due to phoneme distortions and high variability. While self-supervised ASR models like Wav2Vec, HuBERT, and Whisper have shown promise, their effectiveness in dysarthric speech remains unclear. This study systematically benchmarks these models with different decoding strategies, including CTC, seq2seq, and LLM-enhanced decoding (BART,GPT-2, Vicuna). Our contributions include (1) benchmarking ASR architectures for dysarthric speech, (2) introducing LLM-based decoding to improve intelligibility, (3) analyzing generalization across datasets, and (4) providing insights into recognition errors across severity levels. Findings highlight that LLM-enhanced decoding improves dysarthric ASR by leveraging linguistic constraints for phoneme restoration and grammatical correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08001v1">Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Rui Yao, Qi Chai, and Jinhai Yao contributed equally to this work. Corresponding authors: Qi Zhang (zhang.qi@sjtu.edu.cn) and Hao Wang (haowang@hkust-gz.edu.cn)
    </div>
    <details class="paper-abstract">
      "Fedspeak", the stylized and often nuanced language used by the U.S. Federal Reserve, encodes implicit policy signals and strategic stances. The Federal Open Market Committee strategically employs Fedspeak as a communication tool to shape market expectations and influence both domestic and global economic conditions. As such, automatically parsing and interpreting Fedspeak presents a high-impact challenge, with significant implications for financial forecasting, algorithmic trading, and data-driven policy analysis. In this paper, we propose an LLM-based, uncertainty-aware framework for deciphering Fedspeak and classifying its underlying monetary policy stance. Technically, to enrich the semantic and contextual representation of Fedspeak texts, we incorporate domain-specific reasoning grounded in the monetary policy transmission mechanism. We further introduce a dynamic uncertainty decoding module to assess the confidence of model predictions, thereby enhancing both classification accuracy and model reliability. Experimental results demonstrate that our framework achieves state-of-the-art performance on the policy stance analysis task. Moreover, statistical analysis reveals a significant positive correlation between perceptual uncertainty and model error rates, validating the effectiveness of perceptual uncertainty as a diagnostic signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17001v2">PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Personalizing language models by effectively incorporating user interaction history remains a central challenge in the development of adaptive AI systems. While large language models (LLMs) combined with Retrieval-Augmented Generation (RAG) have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graphs, automatically constructed and updated by the LLM itself, and capable of encoding information in multiple formats-including nodes, triplets, higher-order propositions, and episodic traces. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyperedges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle propagation, beam search, and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks-TriviaQA, HotpotQA, and DiaASQ-demonstrating that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10434v3">Spotter+GPT: Turning Sign Spottings into Sentences with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted at the 9th Workshop on Sign Language Translation and Avatar Technologies (SLTAT) in ACM International Conference on Intelligent Virtual Agents (IVA`25)
    </div>
    <details class="paper-abstract">
      Sign Language Translation (SLT) is a challenging task that aims to generate spoken language sentences from sign language videos. In this paper, we introduce a lightweight, modular SLT framework, Spotter+GPT, that leverages the power of Large Language Models (LLMs) and avoids heavy end-to-end training. Spotter+GPT breaks down the SLT task into two distinct stages. First, a sign spotter identifies individual signs within the input video. The spotted signs are then passed to an LLM, which transforms them into meaningful spoken language sentences. Spotter+GPT eliminates the requirement for SLT-specific training. This significantly reduces computational costs and time requirements. The source code and pretrained weights of the Spotter are available at https://gitlab.surrey.ac.uk/cogvispublic/sign-spotter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07935v1">SHIELDA: Structured Handling of Exceptions in LLM-Driven Agentic Workflows</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agentic systems are software systems powered by LLMs that autonomously reason, plan, and execute multi-step workflows to achieve human goals, rather than merely executing predefined steps. During execution, these workflows frequently encounter exceptions. Existing exception handling solutions often treat exceptions superficially, failing to trace execution-phase exceptions to their reasoning-phase root causes. Furthermore, their recovery logic is brittle, lacking structured escalation pathways when initial attempts fail. To tackle these challenges, we first present a comprehensive taxonomy of 36 exception types across 12 agent artifacts. Building on this, we propose SHIELDA (Structured Handling of Exceptions in LLM-Driven Agentic Workflows), a modular runtime exception handling framework for LLM agentic workflows. SHIELDA uses an exception classifier to select a predefined exception handling pattern from a handling pattern registry. These patterns are then executed via a structured handling executor, comprising local handling, flow control, and state recovery, to enable phase-aware recovery by linking exceptions to their root causes and facilitating composable strategies. We validate SHIELDA's effectiveness through a case study on the AutoPR agent, demonstrating effective, cross-phase recovery from a reasoning-induced exception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17066v3">Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02233v2">A Methodological Framework for LLM-Based Mining of Software Repositories</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software engineering research, offering new opportunities for automating repository mining tasks. However, despite their growing popularity, the methodological integration of LLMs into Mining Software Repositories (MSR) remains poorly understood. Existing studies tend to focus on specific capabilities or performance benchmarks, providing limited insight into how researchers utilize LLMs across the full research pipeline. To address this gap, we conduct a mixed-method study that combines a rapid review and questionnaire survey in the field of LLM4MSR. We investigate (1) the approaches and (2) the threats that affect the empirical rigor of researchers involved in this field. Our findings reveal 15 methodological approaches, nine main threats, and 25 mitigation strategies. Building on these findings, we present PRIMES 2.0, a refined empirical framework organized into six stages, comprising 23 methodological substeps, each mapped to specific threats and corresponding mitigation strategies, providing prescriptive and adaptive support throughout the lifecycle of LLM-based MSR studies. Our work contributes to establishing a more transparent and reproducible foundation for LLM-based MSR research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07885v1">Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06225v2">Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the Overconfidence Phenomenon in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce TH-Score, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose LLM-as-a-Fuser, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07849v1">LLMs for Law: Evaluating Legal-Specific LLMs on Contract Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Under review. 4 pages + references
    </div>
    <details class="paper-abstract">
      Despite advances in legal NLP, no comprehensive evaluation covering multiple legal-specific LLMs currently exists for contract classification tasks in contract understanding. To address this gap, we present an evaluation of 10 legal-specific LLMs on three English language contract understanding tasks and compare them with 7 general-purpose LLMs. The results show that legal-specific LLMs consistently outperform general-purpose models, especially on tasks requiring nuanced legal understanding. Legal-BERT and Contracts-BERT establish new SOTAs on two of the three tasks, despite having 69% fewer parameters than the best-performing general-purpose LLM. We also identify CaseLaw-BERT and LexLM as strong additional baselines for contract understanding. Our results provide a holistic evaluation of legal-specific LLMs and will facilitate the development of more accurate contract understanding systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13380v3">DAGR: Decomposition Augmented Graph Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many Natural Language Processing (NLP) tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To address this challenge, we introduce DAGR, a retrieval method that leverages both complex questions and their decomposition in subquestions to extract relevant, linked textual subgraphs. DAGR first breaks down complex queries, retrieves subgraphs guided by a weighted similarity function over both the original and decomposed queries, and creates a question-specific knowledge graph to guide answer generation. The resulting Graph-RAG pipeline is suited to handle complex multi-hop questions and effectively reason over graph-structured data. We evaluate DAGR on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07583v3">Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at https://github.com/deep-spin/treqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07805v1">Can You Trick the Grader? Adversarial Persuasion of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      As large language models take on growing roles as automated evaluators in practical settings, a critical question arises: Can individuals persuade an LLM judge to assign unfairly high scores? This study is the first to reveal that strategically embedded persuasive language can bias LLM judges when scoring mathematical reasoning tasks, where correctness should be independent of stylistic variation. Grounded in Aristotle's rhetorical principles, we formalize seven persuasion techniques (Majority, Consistency, Flattery, Reciprocity, Pity, Authority, Identity) and embed them into otherwise identical responses. Across six math benchmarks, we find that persuasive language leads LLM judges to assign inflated scores to incorrect solutions, by up to 8% on average, with Consistency causing the most severe distortion. Notably, increasing model size does not substantially mitigate this vulnerability. Further analysis demonstrates that combining multiple persuasion techniques amplifies the bias, and pairwise evaluation is likewise susceptible. Moreover, the persuasive effect persists under counter prompting strategies, highlighting a critical vulnerability in LLM-as-a-Judge pipelines and underscoring the need for robust defenses against persuasion-based attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07785v1">Grove MoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The Mixture of Experts (MoE) architecture is a cornerstone of modern state-of-the-art (SOTA) large language models (LLMs). MoE models facilitate scalability by enabling sparse parameter activation. However, traditional MoE architecture uses homogeneous experts of a uniform size, activating a fixed number of parameters irrespective of input complexity and thus limiting computational efficiency. To overcome this limitation, we introduce Grove MoE, a novel architecture incorporating experts of varying sizes, inspired by the heterogeneous big.LITTLE CPU architecture. This architecture features novel adjugate experts with a dynamic activation mechanism, enabling model capacity expansion while maintaining manageable computational overhead. Building on this architecture, we present GroveMoE-Base and GroveMoE-Inst, 33B-parameter LLMs developed by applying an upcycling strategy to the Qwen3-30B-A3B-Base model during mid-training and post-training. GroveMoE models dynamically activate 3.14-3.28B parameters based on token complexity and achieve performance comparable to SOTA open-source models of similar or even larger size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07781v1">SASST: Leveraging Syntax-Aware Chunking and LLMs for Simultaneous Speech Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This work proposes a grammar-based chunking strategy that segments input streams into semantically complete units by parsing dependency relations (e.g., noun phrase boundaries, verb-object structures) and punctuation features. The method ensures chunk coherence and minimizes semantic fragmentation. Building on this mechanism, we present SASST (Syntax-Aware Simultaneous Speech Translation), an end-to-end framework integrating frozen Whisper encoder and decoder-only LLM. The unified architecture dynamically outputs translation tokens or <WAIT> symbols to jointly optimize translation timing and content, with target-side reordering addressing word-order divergence. Experiments on CoVoST2 multilingual corpus En-{De, Zh, Ja} demonstrate significant translation quality improvements across languages and validate the effectiveness of syntactic structures in LLM-driven SimulST systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07329v1">Efficient Edge LLMs Deployment via HessianAware Quantization and CPU GPU Collaborative</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      With the breakthrough progress of large language models (LLMs) in natural language processing and multimodal tasks, efficiently deploying them on resource-constrained edge devices has become a critical challenge. The Mixture of Experts (MoE) architecture enhances model capacity through sparse activation, but faces two major difficulties in practical deployment: (1) The presence of numerous outliers in activation distributions leads to severe degradation in quantization accuracy for both activations and weights, significantly impairing inference performance; (2) Under limited memory, efficient offloading and collaborative inference of expert modules struggle to balance latency and throughput. To address these issues, this paper proposes an efficient MoE edge deployment scheme based on Hessian-Aware Quantization (HAQ) and CPU-GPU collaborative inference. First, by introducing smoothed Hessian matrix quantization, we achieve joint 8-bit quantization of activations and weights, which significantly alleviates the accuracy loss caused by outliers while ensuring efficient implementation on mainstream hardware. Second, we design an expert-level collaborative offloading and inference mechanism, which, combined with expert activation path statistics, enables efficient deployment and scheduling of expert modules between CPU and GPU, greatly reducing memory footprint and inference latency. Extensive experiments validate the effectiveness of our method on mainstream large models such as the OPT series and Mixtral 8*7B: on datasets like Wikitext2 and C4, the inference accuracy of the low-bit quantized model approaches that of the full-precision model, while GPU memory usage is reduced by about 60%, and inference latency is significantly improved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07321v1">ObfusQAte: A Proposed Framework to Evaluate LLM Robustness on Obfuscated Factual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has significantly contributed to the development of equitable AI systems capable of factual question-answering (QA). However, no known study tests the LLMs' robustness when presented with obfuscated versions of questions. To systematically evaluate these limitations, we propose a novel technique, ObfusQAte and, leveraging the same, introduce ObfusQA, a comprehensive, first of its kind, framework with multi-tiered obfuscation levels designed to examine LLM capabilities across three distinct dimensions: (i) Named-Entity Indirection, (ii) Distractor Indirection, and (iii) Contextual Overload. By capturing these fine-grained distinctions in language, ObfusQA provides a comprehensive benchmark for evaluating LLM robustness and adaptability. Our study observes that LLMs exhibit a tendency to fail or generate hallucinated responses when confronted with these increasingly nuanced variations. To foster research in this direction, we make ObfusQAte publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04903v2">RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07252v1">Tasa: Thermal-aware 3D-Stacked Architecture Design with Bandwidth Sharing for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted by ICCAD'2025
    </div>
    <details class="paper-abstract">
      The autoregressive decoding in LLMs is the major inference bottleneck due to the memory-intensive operations and limited hardware bandwidth. 3D-stacked architecture is a promising solution with significantly improved memory bandwidth, which vertically stacked multi DRAM dies on top of logic die. However, our experiments also show the 3D-stacked architecture faces severer thermal issues compared to 2D architecture, in terms of thermal temperature, gradient and scalability. To better exploit the potential of 3D-stacked architecture, we present Tasa, a heterogeneous architecture with cross-stack thermal optimizations to balance the temperature distribution and maximize the performance under the thermal constraints. High-performance core is designed for compute-intensive operations, while high-efficiency core is used for memory-intensive operators, e.g. attention layers. Furthermore, we propose a bandwidth sharing scheduling to improve the bandwidth utilization in such heterogeneous architecture. Extensive thermal experiments show that our Tasa architecture demonstrates greater scalability compared with the homogeneous 3D-stacked architecture, i.e. up to 5.55 $\tccentigrade$, 9.37 $\tccentigrade$, and 7.91 $\tccentigrade$ peak temperature reduction for 48, 60, and 72 core configurations. Our experimental for Llama-65B and GPT-3 66B inferences also demonstrate 2.85x and 2.21x speedup are obtained over the GPU baselines and state-of-the-art heterogeneous PIM-based LLM accelerator
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07227v1">LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted by ICCAD'2025
    </div>
    <details class="paper-abstract">
      LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) techniques have been explored at the algorithmic and hardware levels. However, speculative inference results in more compute-intensive GEMM operations, creating new design trade-offs for existing GEMV-accelerated PIM architectures. Furthermore, there exists a significant amount of redundant draft tokens in tree-based speculative inference, necessitating efficient token management schemes to minimize energy consumption. In this work, we present LP-Spec, an architecture-dataflow co-design leveraging hybrid LPDDR5 performance-enhanced PIM architecture with draft token pruning and dynamic workload scheduling to accelerate LLM speculative inference. A near-data memory controller is proposed to enable data reallocation between DRAM and PIM banks. Furthermore, a data allocation unit based on the hardware-aware draft token pruner is developed to minimize energy consumption and fully exploit parallel execution opportunities. Compared to end-to-end LLM inference on other mobile solutions such as mobile NPUs or GEMV-accelerated PIMs, our LP-Spec achieves 13.21x, 7.56x, and 99.87x improvements in performance, energy efficiency, and energy-delay-product (EDP). Compared with prior AttAcc PIM and RTX 3090 GPU, LP-Spec can obtain 12.83x and 415.31x EDP reduction benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07221v1">LLM-based Agents for Automated Confounder Discovery and Subgroup Analysis in Causal Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Estimating individualized treatment effects from observational data presents a persistent challenge due to unmeasured confounding and structural bias. Causal Machine Learning (causal ML) methods, such as causal trees and doubly robust estimators, provide tools for estimating conditional average treatment effects. These methods have limited effectiveness in complex real-world environments due to the presence of latent confounders or those described in unstructured formats. Moreover, reliance on domain experts for confounder identification and rule interpretation introduces high annotation cost and scalability concerns. In this work, we proposed Large Language Model-based agents for automated confounder discovery and subgroup analysis that integrate agents into the causal ML pipeline to simulate domain expertise. Our framework systematically performs subgroup identification and confounding structure discovery by leveraging the reasoning capabilities of LLM-based agents, which reduces human dependency while preserving interpretability. Experiments on real-world medical datasets show that our proposed approach enhances treatment effect estimation robustness by narrowing confidence intervals and uncovering unrecognized confounding biases. Our findings suggest that LLM-based agents offer a promising path toward scalable, trustworthy, and semantically aware causal inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19115v2">FP4 All the Way: Fully Quantized Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      We demonstrate, for the first time, fully quantized training (FQT) of large language models (LLMs) using predominantly 4-bit floating-point (FP4) precision for weights, activations, and gradients on datasets up to 200 billion tokens. We extensively investigate key design choices for FP4, including block sizes, scaling formats, and rounding methods. Our analysis shows that the NVFP4 format, where each block of 16 FP4 values (E2M1) shares a scale represented in E4M3, provides optimal results. We use stochastic rounding for backward and update passes and round-to-nearest for the forward pass to enhance stability. Additionally, we identify a theoretical and empirical threshold for effective quantized training: when the gradient norm falls below approximately $\sqrt{3}$ times the quantization noise, quantized training becomes less effective. Leveraging these insights, we successfully train a 7-billion-parameter model on 256 Intel Gaudi2 accelerators. The resulting FP4-trained model achieves downstream task performance comparable to a standard BF16 baseline, confirming that FP4 training is a practical and highly efficient approach for large-scale LLM training. A reference implementation is supplied in https://github.com/Anonymous1252022/fp4-all-the-way .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08780v3">How Relevance Emerges: Interpreting LoRA Fine-Tuning in Reranking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Extended Abstract
    </div>
    <details class="paper-abstract">
      We conduct a behavioral exploration of LoRA fine-tuned LLMs for Passage Reranking to understand how relevance signals are learned and deployed by Large Language Models. By fine-tuning Mistral-7B, LLaMA3.1-8B, and Pythia-6.9B on MS MARCO under diverse LoRA configurations, we investigate how relevance modeling evolves across checkpoints, the impact of LoRA rank (1, 2, 8, 32), and the relative importance of updated MHA vs. MLP components. Our ablations reveal which layers and projections within LoRA transformations are most critical for reranking accuracy. These findings offer fresh explanations into LoRA's adaptation mechanisms, setting the stage for deeper mechanistic studies in Information Retrieval. All models used in this study have been shared.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07195v1">Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Semantic Alignment</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated impressive capabilities in natural language processing due to their strong generalization and sequence modeling capabilities. However, their direct application to time series forecasting remains challenging due to two fundamental issues: the inherent heterogeneity of temporal patterns and the modality gap between continuous numerical signals and discrete language representations. In this work, we propose TALON, a unified framework that enhances LLM-based forecasting by modeling temporal heterogeneity and enforcing semantic alignment. Specifically, we design a Heterogeneous Temporal Encoder that partitions multivariate time series into structurally coherent segments, enabling localized expert modeling across diverse temporal patterns. To bridge the modality gap, we introduce a Semantic Alignment Module that aligns temporal features with LLM-compatible representations, enabling effective integration of time series into language-based models while eliminating the need for handcrafted prompts during inference. Extensive experiments on seven real-world benchmarks demonstrate that TALON achieves superior performance across all datasets, with average MSE improvements of up to 11\% over recent state-of-the-art methods. These results underscore the effectiveness of incorporating both pattern-aware and semantic-aware designs when adapting LLMs for time series forecasting. The code is available at: https://github.com/syrGitHub/TALON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23842v2">Document Valuation in LLM Summaries: A Cluster Shapley Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these models enhance user experience by generating coherent summaries, they obscure the contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries. We propose using Shapley values, a game-theoretic method that allocates credit based on each document's marginal contribution. Although theoretically appealing, Shapley values are expensive to compute at scale. We therefore propose Cluster Shapley, an efficient approximation algorithm that leverages semantic similarity between documents. By clustering documents using LLM-based embeddings and computing Shapley values at the cluster level, our method significantly reduces computation while maintaining attribution quality. We demonstrate our approach to a summarization task using Amazon product reviews. Cluster Shapley significantly reduces computational complexity while maintaining high accuracy, outperforming baseline methods such as Monte Carlo sampling and Kernel SHAP with a better efficient frontier. Our approach is agnostic to the exact LLM used, the summarization process used, and the evaluation procedure, which makes it broadly applicable to a variety of summarization settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07572v3">WebWalker: Benchmarking LLMs in Web Traversal</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) demonstrates remarkable performance across tasks in open-domain question-answering. However, traditional search engines may retrieve shallow content, limiting the ability of LLMs to handle complex, multi-layered information. To address it, we introduce WebWalkerQA, a benchmark designed to assess the ability of LLMs to perform web traversal. It evaluates the capacity of LLMs to traverse a website's subpages to extract high-quality data systematically. We propose WebWalker, which is a multi-agent framework that mimics human-like web navigation through an explore-critic paradigm. Extensive experimental results show that WebWalkerQA is challenging and demonstrates the effectiveness of RAG combined with WebWalker, through the horizontal and vertical integration in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07172v1">Gradient Surgery for Safe LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Fine-tuning-as-a-Service introduces a critical vulnerability where a few malicious examples mixed into the user's fine-tuning dataset can compromise the safety alignment of Large Language Models (LLMs). While a recognized paradigm frames safe fine-tuning as a multi-objective optimization problem balancing user task performance with safety alignment, we find existing solutions are critically sensitive to the harmful ratio, with defenses degrading sharply as harmful ratio increases. We diagnose that this failure stems from conflicting gradients, where the user-task update directly undermines the safety objective. To resolve this, we propose SafeGrad, a novel method that employs gradient surgery. When a conflict is detected, SafeGrad nullifies the harmful component of the user-task gradient by projecting it onto the orthogonal plane of the alignment gradient, allowing the model to learn the user's task without sacrificing safety. To further enhance robustness and data efficiency, we employ a KL-divergence alignment loss that learns the rich, distributional safety profile of the well-aligned foundation model. Extensive experiments show that SafeGrad provides state-of-the-art defense across various LLMs and datasets, maintaining robust safety even at high harmful ratios without compromising task fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v2">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [Accepted]
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08332v1">Energy-Aware Code Generation with LLMs: Benchmarking Small vs. Large Language Models for Sustainable AI Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for code generation. However, commercial models like ChatGPT require significant computing power, which leads to high energy use and carbon emissions. This has raised concerns about their environmental impact. In this study, we evaluate open-source Small Language Models (SLMs) trained explicitly for code generation and compare their performance and energy efficiency against large LLMs and efficient human-written Python code. The goal is to investigate whether SLMs can match the performance of LLMs on certain types of programming problems while producing more energy-efficient code. We evaluate 150 coding problems from LeetCode, evenly distributed across three difficulty levels: easy, medium, and hard. Our comparison includes three small open-source models, StableCode-3B, StarCoderBase-3B, and Qwen2.5-Coder-3B-Instruct, and two large commercial models, GPT-4.0 and DeepSeek-Reasoner. The generated code is evaluated using four key metrics: run-time, memory usage, energy consumption, and correctness. We use human-written solutions as a baseline to assess the quality and efficiency of the model-generated code. Results indicate that LLMs achieve the highest correctness across all difficulty levels, but SLMs are often more energy-efficient when their outputs are correct. In over 52% of the evaluated problems, SLMs consumed the same or less energy than LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09208v1">CoMoE: Collaborative Optimization of Expert Aggregation and Offloading for MoE-based LLMs at Edge</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has driven the adoption of Mixture-of-Experts (MoE) architectures as a promising solution to scale model capacity while controlling computational costs. However, deploying MoE models in resource-constrained mobile edge computing environments presents significant challenges due to their large memory footprint and dynamic expert activation patterns. To address these challenges, we propose a novel dynamic resource-aware collaborative optimization framework that jointly optimizes expert aggregation granularity and offloading strategies based on real-time device resource states, network conditions, and input characteristics in mobile edge environments, denoted as CoMoE. In CoMoE, we first systematically analyze existing expert aggregation techniques, including expert parameter merging,knowledge distillation,and parameter sharing decomposition, identifying their limitations in dynamic mobile environments.We then investigate expert offloading strategies encompassing expert prediction and prefetching, expert caching and scheduling, and multi-tier storage architectures, revealing the interdependencies between routing decisions and offloading performance.The CoMoE incorporates adaptive scheduling mechanisms that respond to user mobility and varying network conditions, enabling efficient MoE deployment across heterogeneous edge devices. Extensive experiments on real mobile edge testbeds demonstrate that CoMoE achieves approximately 70% reduction in memory usage compared to baseline methods, 10.5% lower inference latency than existing expert offloading techniques, while maintaining model performance stability. For large-scale MoE models (e.g,7.4B-parameter Switch-Base-128), the CoMoE reduces memory requirements from 15.6GB to 4.7GB, enabling deployment on resource-constrained mobile edge devices that previously could only support much smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09652v2">How Chinese are Chinese Language Models? The Puzzling Lack of Language Policy in China's LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 We have reworked the paper substantially. Please refer to the new, updated article: arXiv:2504.00289
    </div>
    <details class="paper-abstract">
      Contemporary language models are increasingly multilingual, but Chinese LLM developers must navigate complex political and business considerations of language diversity. Language policy in China aims at influencing the public discourse and governing a multi-ethnic society, and has gradually transitioned from a pluralist to a more assimilationist approach since 1949. We explore the impact of these influences on current language technology. We evaluate six open-source multilingual LLMs pre-trained by Chinese companies on 18 languages, spanning a wide range of Chinese, Asian, and Anglo-European languages. Our experiments show Chinese LLMs performance on diverse languages is indistinguishable from international LLMs. Similarly, the models' technical reports also show lack of consideration for pretraining data language coverage except for English and Mandarin Chinese. Examining Chinese AI policy, model experiments, and technical reports, we find no sign of any consistent policy, either for or against, language diversity in China's LLM development. This leaves a puzzling fact that while China regulates both the languages people use daily as well as language model development, they do not seem to have any policy on the languages in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07466v1">Grounding Natural Language for Multi-agent Decision-Making with Multi-agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Language is a ubiquitous tool that is foundational to reasoning and collaboration, ranging from everyday interactions to sophisticated problem-solving tasks. The establishment of a common language can serve as a powerful asset in ensuring clear communication and understanding amongst agents, facilitating desired coordination and strategies. In this work, we extend the capabilities of large language models (LLMs) by integrating them with advancements in multi-agent decision-making algorithms. We propose a systematic framework for the design of multi-agentic large language models (LLMs), focusing on key integration practices. These include advanced prompt engineering techniques, the development of effective memory architectures, multi-modal information processing, and alignment strategies through fine-tuning algorithms. We evaluate these design choices through extensive ablation studies on classic game settings with significant underlying social dilemmas and game-theoretic considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07434v1">Let's Revise Step-by-Step: A Unified Local Search Framework for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with inference-time scaling techniques show promise for code generation, yet face notable efficiency and scalability challenges. Construction-based tree-search methods suffer from rapid growth in tree size, high token consumption, and lack of anytime property. In contrast, improvement-based methods offer better performance but often struggle with uninformative reward signals and inefficient search strategies. In this work, we propose \textbf{ReLoc}, a unified local search framework which effectively performs step-by-step code revision. Specifically, ReLoc explores a series of local revisions through four key algorithmic components: initial code drafting, neighborhood code generation, candidate evaluation, and incumbent code updating, each of which can be instantiated with specific decision rules to realize different local search algorithms such as Hill Climbing (HC) or Genetic Algorithm (GA). Furthermore, we develop a specialized revision reward model that evaluates code quality based on revision distance to produce fine-grained preferences that guide the local search toward more promising candidates. Finally, our extensive experimental results demonstrate that our approach achieves superior performance across diverse code generation tasks, significantly outperforming both construction-based tree search as well as the state-of-the-art improvement-based code generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07078v3">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07421v1">Triple-S: A Collaborative Multi-LLM Framework for Solving Long-Horizon Implicative Tasks in Robotics</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted to IROS 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) to write policy code for controlling robots has gained significant attention. However, in long-horizon implicative tasks, this approach often results in API parameter, comments and sequencing errors, leading to task failure. To address this problem, we propose a collaborative Triple-S framework that involves multiple LLMs. Through In-Context Learning, different LLMs assume specific roles in a closed-loop Simplification-Solution-Summary process, effectively improving success rates and robustness in long-horizon implicative tasks. Additionally, a novel demonstration library update mechanism which learned from success allows it to generalize to previously failed tasks. We validate the framework in the Long-horizon Desktop Implicative Placement (LDIP) dataset across various baseline models, where Triple-S successfully executes 89% of tasks in both observable and partially observable scenarios. Experiments in both simulation and real-world robot settings further validated the effectiveness of Triple-S. Our code and dataset is available at: https://github.com/Ghbbbbb/Triple-S.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06327v3">A Taxonomy of Inefficiencies in LLM-Generated Python Code</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for automated code generation with promising results. Although prior research has assessed LLM-generated code and identified various quality issues -- such as redundancy, poor maintainability, and sub-optimal performance a systematic understanding and categorization of these inefficiencies remain unexplored. Without such knowledge, practitioners struggle to optimize LLM-generated code for real-world applications, limiting its adoption. This study can also guide improving code LLMs, enhancing the quality and efficiency of code generation. Therefore, in this study, we empirically investigate inefficiencies in LLM-generated code by state-of-the-art models, i.e., CodeLlama, DeepSeek-Coder, and CodeGemma. To do so, we analyze 492 generated code snippets in the HumanEval++ dataset. We then construct a taxonomy of inefficiencies in LLM-generated code that includes 5 categories General Logic, Performance, Readability, Maintainability, and Errors) and 19 subcategories of inefficiencies. We then validate the proposed taxonomy through an online survey with 58 LLM practitioners and researchers. Our study indicates that logic and performance-related inefficiencies are the most popular, relevant, and frequently co-occur and impact overall code quality inefficiency. Our taxonomy provides a structured basis for evaluating the quality LLM-generated code and guiding future research to improve code generation efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07408v1">Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 16 pages, 12 figures, accepted at ICML 2025 New in ML Workshop
    </div>
    <details class="paper-abstract">
      In this study, we wish to showcase the unique utility of large language models (LLMs) in financial semantic annotation and alpha signal discovery. Leveraging a corpus of company-related tweets, we use an LLM to automatically assign multi-label event categories to high-sentiment-intensity tweets. We align these labeled sentiment signals with forward returns over 1-to-7-day horizons to evaluate their statistical efficacy and market tradability. Our experiments reveal that certain event labels consistently yield negative alpha, with Sharpe ratios as low as -0.38 and information coefficients exceeding 0.05, all statistically significant at the 95\% confidence level. This study establishes the feasibility of transforming unstructured social media text into structured, multi-label event variables. A key contribution of this work is its commitment to transparency and reproducibility; all code and methodologies are made publicly available. Our results provide compelling evidence that social media sentiment is a valuable, albeit noisy, signal in financial forecasting and underscore the potential of open-source frameworks to democratize algorithmic trading research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07371v1">AutoAssert 1: A LoRA Fine-Tuned LLM Model for Efficient Automated Assertion Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 16pages,6figures
    </div>
    <details class="paper-abstract">
      As the complexity of software systems continues to increase, the demand for automated testing and maintenance tools is growing exponentially. To meet this urgent need, we propose a new assertion generation method based on Hardware Description Language (HDL). This method combines a lightweight, parameter-adjustable large language model (LLM) with the Unsloth platform to automatically generate test cases, thereby significantly reducing training costs without sacrificing accuracy or generalization performance. Empirical evaluation shows that our method can efficiently generate assertions that strictly conform to the hardware logic. This framework provides a robust and flexible solution to modern software testing and maintenance challenges. https://github.com/liusu-orange/AutoAssert-1 and https://gitee.com/OpenBPU/auto-assert1 are the locations of the source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v3">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09426v4">FlatQuant: Flatness Matters for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 27 pages, accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Recently, quantization has been widely used for the compression and acceleration of large language models (LLMs). Due to the outliers in LLMs, it is crucial to flatten weights and activations to minimize quantization error with equally spaced quantization points. Prior research explores various pre-quantization transformations to suppress outliers, such as per-channel scaling and Hadamard transformation. However, we observe that these transformed weights and activations can still exhibit steep and dispersed distributions. In this paper, we propose FlatQuant (Fast and Learnable Affine Transformation), a new post-training quantization approach that enhances the flatness of weights and activations. Our approach identifies optimal affine transformations for each linear layer, calibrated in hours via a lightweight objective. To reduce runtime overhead of affine transformation, we apply Kronecker product with two lightweight matrices, and fuse all operations in FlatQuant into a single kernel. Extensive experiments demonstrate that FlatQuant establishes a new state-of-the-art benchmark for quantization. For example, it achieves less than 1\% accuracy drop for W4A4 quantization on the LLaMA-3-70B model, surpassing SpinQuant by 7.5\%. Additionally, it provides up to 2.3x prefill speedup and 1.7x decoding speedup compared to the FP16 model. Code is available at: https://github.com/ruikangliu/FlatQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07353v1">Rethinking Domain-Specific LLM Benchmark Construction: A Comprehensiveness-Compactness Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Numerous benchmarks have been built to evaluate the domain-specific abilities of large language models (LLMs), highlighting the need for effective and efficient benchmark construction. Existing domain-specific benchmarks primarily focus on the scaling law, relying on massive corpora for supervised fine-tuning or generating extensive question sets for broad coverage. However, the impact of corpus and question-answer (QA) set design on the precision and recall of domain-specific LLMs remains unexplored. In this paper, we address this gap and demonstrate that the scaling law is not always the optimal principle for benchmark construction in specific domains. Instead, we propose Comp-Comp, an iterative benchmarking framework based on a comprehensiveness-compactness principle. Here, comprehensiveness ensures semantic recall of the domain, while compactness enhances precision, guiding both corpus and QA set construction. To validate our framework, we conducted a case study in a well-renowned university, resulting in the creation of XUBench, a large-scale and comprehensive closed-domain benchmark. Although we use the academic domain as the case in this work, our Comp-Comp framework is designed to be extensible beyond academia, providing valuable insights for benchmark construction across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07117v1">From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 18 pages, 3 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) have emerged as powerful tools for learning over structured data, including text-attributed graphs, which are common in domains such as citation networks, social platforms, and knowledge graphs. GNNs are not inherently interpretable and thus, many explanation methods have been proposed. However, existing explanation methods often struggle to generate interpretable, fine-grained rationales, especially when node attributes include rich natural language. In this work, we introduce LOGIC, a lightweight, post-hoc framework that uses large language models (LLMs) to generate faithful and interpretable explanations for GNN predictions. LOGIC projects GNN node embeddings into the LLM embedding space and constructs hybrid prompts that interleave soft prompts with textual inputs from the graph structure. This enables the LLM to reason about GNN internal representations and produce natural language explanations along with concise explanation subgraphs. Our experiments across four real-world TAG datasets demonstrate that LOGIC achieves a favorable trade-off between fidelity and sparsity, while significantly improving human-centric metrics such as insightfulness. LOGIC sets a new direction for LLM-based explainability in graph learning by aligning GNN internals with human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14617v2">SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 25 pages, 16 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Global cloud service providers handle inference workloads for Large Language Models (LLMs) that span latency-sensitive (e.g., chatbots) and insensitive (e.g., report writing) tasks, resulting in diverse and often conflicting Service Level Agreement (SLA) requirements. Managing such mixed workloads is challenging due to the complexity of the inference serving stack, which encompasses multiple models, GPU hardware, and global data centers. Existing solutions often silo such fast and slow tasks onto separate GPU resource pools with different SLAs, but this leads to significant under-utilization of expensive accelerators due to load mismatch. In this article, we characterize the LLM serving workloads at Microsoft Office 365, one of the largest users of LLMs within Microsoft Azure cloud with over 10 million requests per day, and highlight key observations across workloads in different data center regions and across time. This is one of the first such public studies of Internet-scale LLM workloads. We use these insights to propose SageServe, a comprehensive LLM serving framework that dynamically adapts to workload demands using multi-timescale control knobs. It combines short-term request routing to data centers with long-term scaling of GPU VMs and model placement with higher lead times, and co-optimizes the routing and resource allocation problem using a traffic forecast model and an Integer Linear Programming (ILP) solution. We evaluate SageServe through real runs and realistic simulations on 10 million production requests across three regions and four open-source models. We achieve up to 25% savings in GPU-hours compared to the current baseline deployment and reduce GPU-hour wastage due to inefficient auto-scaling by 80%, resulting in a potential monthly cost savings of up to $2.5 million, while maintaining tail latency and meeting SLAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07075v1">Surgical Knowledge Rewrite in Compact LLMs: An 'Unlearn-then-Learn' Strategy with ($IA^3$) for Localized Factual Modulation and Catastrophic Forgetting Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 9 pages, 2 visual aids
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with dynamic knowledge updates, especially when new information conflicts with deeply embedded facts. Such conflicting factual edits often lead to two critical issues: resistance to adopting the new fact and severe catastrophic forgetting of unrelated knowledge. This paper introduces and evaluates a novel "unlearn-then-learn" strategy for precise knowledge editing in LLMs, leveraging the parameter-efficient fine-tuning (PEFT) technique, Infused Adapter by Inhibiting and Amplifying Inner Activations ($IA^3$). Crucially, this two-stage approach is powered by an initial circuit localization phase that identifies and targets the specific internal components responsible for encoding the conflicting fact. Through a rigorous experimental methodology on microsoft/Phi-3-mini-4k-instruct, we demonstrate that this mechanistically informed two-stage approach achieves near-perfect accuracy (98.50%) for the new, modulated fact while simultaneously effectively suppressing the original conflicting fact (96.00% forget rate). Critically, our strategy exhibits unprecedented localization (72.00% F_control accuracy), dramatically mitigating catastrophic forgetting observed in direct fine-tuning approaches (which showed as low as ~20% F_control accuracy), a direct benefit of our targeted interpretability-guided intervention. Furthermore, qualitative analysis reveals a nuanced mechanism of "soft forgetting," where original knowledge is suppressed from default retrieval but remains latent and conditionally accessible, enhancing model safety and control. These findings represent a significant advancement towards precise, localized, and safe knowledge management in compact LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v3">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 18 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07063v1">Towards Safer AI Moderation: Evaluating LLM Moderators Through a Unified Benchmark Dataset and Advocating a Human-First Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      As AI systems become more integrated into daily life, the need for safer and more reliable moderation has never been greater. Large Language Models (LLMs) have demonstrated remarkable capabilities, surpassing earlier models in complexity and performance. Their evaluation across diverse tasks has consistently showcased their potential, enabling the development of adaptive and personalized agents. However, despite these advancements, LLMs remain prone to errors, particularly in areas requiring nuanced moral reasoning. They struggle with detecting implicit hate, offensive language, and gender biases due to the subjective and context-dependent nature of these issues. Moreover, their reliance on training data can inadvertently reinforce societal biases, leading to inconsistencies and ethical concerns in their outputs. To explore the limitations of LLMs in this role, we developed an experimental framework based on state-of-the-art (SOTA) models to assess human emotions and offensive behaviors. The framework introduces a unified benchmark dataset encompassing 49 distinct categories spanning the wide spectrum of human emotions, offensive and hateful text, and gender and racial biases. Furthermore, we introduced SafePhi, a QLoRA fine-tuned version of Phi-4, adapting diverse ethical contexts and outperforming benchmark moderators by achieving a Macro F1 score of 0.89, where OpenAI Moderator and Llama Guard score 0.77 and 0.74, respectively. This research also highlights the critical domains where LLM moderators consistently underperformed, pressing the need to incorporate more heterogeneous and representative data with human-in-the-loop, for better model robustness and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07054v1">Membership and Memorization in LLM Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Recent advances in Knowledge Distillation (KD) aim to mitigate the high computational demands of Large Language Models (LLMs) by transferring knowledge from a large ''teacher'' to a smaller ''student'' model. However, students may inherit the teacher's privacy when the teacher is trained on private data. In this work, we systematically characterize and investigate membership and memorization privacy risks inherent in six LLM KD techniques. Using instruction-tuning settings that span seven NLP tasks, together with three teacher model families (GPT-2, LLAMA-2, and OPT), and various size student models, we demonstrate that all existing LLM KD approaches carry membership and memorization privacy risks from the teacher to its students. However, the extent of privacy risks varies across different KD techniques. We systematically analyse how key LLM KD components (KD objective functions, student training data and NLP tasks) impact such privacy risks. We also demonstrate a significant disagreement between memorization and membership privacy risks of LLM KD techniques. Finally, we characterize per-block privacy risk and demonstrate that the privacy risk varies across different blocks by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22134v2">IntentFlow: Interactive Support for Communicating Intent with LLMs in Writing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are widely used for writing, users often struggle to express their nuanced and evolving intents through prompt-based interfaces. Intents -- low-level strategies or preferences for achieving a writing goal -- are often vague, fluid, or even subconscious, making it difficult for users to articulate and adjust them. To address this, we present IntentFlow, which supports the communication of dynamically evolving intents throughout LLM-assisted writing. IntentFlow extracts goals and intents from user prompts and presents them as editable interface components, which users can revise, remove, or refine via direct manipulation or follow-up prompts. Visual links connect each component to the output segments it influences, helping users understand model behavior. In a within-subjects study (N=12), participants using IntentFlow, compared to a chat-based baseline, expressed their intents more easily and in detail, engaged in more meaningful actions to communicate intents, such as adjusting and deleting, and produced outputs that better aligned with their evolving intents. We found that editable intent representations help users refine and consolidate a final set of intents, which can be reused across similar tasks to support consistent and transferable LLM-assisted writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06978v1">SSD Offloading for LLM Mixture-of-Experts Weights Considered Harmful in Energy Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 4 pages, 6 figures, accepted at IEEE Computer Architecture Letters
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) applying Mixture-of-Experts (MoE) scale to trillions of parameters but require vast memory, motivating a line of research to offload expert weights from fast-but-small DRAM (HBM) to denser Flash SSDs. While SSDs provide cost-effective capacity, their read energy per bit is substantially higher than that of DRAM. This paper quantitatively analyzes the energy implications of offloading MoE expert weights to SSDs during the critical decode stage of LLM inference. Our analysis, comparing SSD, CPU memory (DDR), and HBM storage scenarios for models like DeepSeek-R1, reveals that offloading MoE weights to current SSDs drastically increases per-token-generation energy consumption (e.g., by up to ~12x compared to the HBM baseline), dominating the total inference energy budget. Although techniques like prefetching effectively hide access latency, they cannot mitigate this fundamental energy penalty. We further explore future technological scaling, finding that the inherent sparsity of MoE models could potentially make SSDs energy-viable if Flash read energy improves significantly, roughly by an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06963v1">MASteer: Multi-Agent Adaptive Steer Strategy for End-to-End LLM Trustworthiness Repair</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face persistent and evolving trustworthiness issues, motivating developers to seek automated and flexible repair methods that enable convenient deployment across diverse scenarios. Existing repair methods like supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) are costly and slow, while prompt engineering lacks robustness and scalability. Representation engineering, which steers model behavior by injecting targeted concept vectors during inference, offers a lightweight, training-free alternative. However, current approaches depend on manually crafted samples and fixed steering strategies, limiting automation and adaptability. To overcome these challenges, we propose MASteer, the first end-to-end framework for trustworthiness repair in LLMs based on representation engineering. MASteer integrates two core components: AutoTester, a multi-agent system that generates diverse, high-quality steer samples tailored to developer needs; and AutoRepairer, which constructs adaptive steering strategies with anchor vectors for automated, context-aware strategy selection during inference. Experiments on standard and customized trustworthiness tasks show MASteer consistently outperforms baselines, improving metrics by 15.36% on LLaMA-3.1-8B-Chat and 4.21% on Qwen-3-8B-Chat, while maintaining general model capabilities. MASteer demonstrates strong robustness, generalization, and practical value for scalable, efficient trustworthiness repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06948v1">Kairos: Low-latency Multi-Agent Serving with Shared LLMs and Excessive Loads in the Public Cloud</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Multi-agent applications utilize the advanced capabilities of large language models (LLMs) for intricate task completion through agent collaboration in a workflow. Under this situation, requests from different agents usually access the same shared LLM to perform different kinds of tasks, forcing the shared LLM to suffer excessive loads. However, existing works have low serving performance for these multi-agent applications, mainly due to the ignorance of inter-agent latency and resource differences for request scheduling. We therefore propose Kairos, a multi-agent orchestration system that optimizes end-to-end latency for multi-agent applications. Kairos consists of a workflow orchestrator, a workflow-aware priority scheduler, and a memory-aware dispatcher. The orchestrator collects agent-specific information for online workflow analysis. The scheduler decides the serving priority of the requests based on their latency characteristics to reduce the overall queuing. The dispatcher dispatches the requests to different LLM instances based on their memory demands to avoid GPU overloading. Experimental results show that Kairos reduces end-to-end latency by 17.8% to 28.4% compared to state-of-the-art works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06944v1">AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of \textbf{implicit rewards}, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce \textbf{Adaptive Meta Fine-Tuning (AMFT)}, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a \textbf{meta-gradient adaptive weight controller} that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. This forward-looking approach, regularized by policy entropy for stability, autonomously discovers an effective training curriculum. We conduct a comprehensive evaluation on challenging benchmarks spanning mathematical reasoning, abstract visual reasoning (General Points), and vision-language navigation (V-IRL). AMFT consistently establishes a new state-of-the-art and demonstrats superior generalization on out-of-distribution (OOD) tasks. Ablation studies and training dynamic analysis confirm that the meta-learning controller is crucial for AMFT's stability, sample efficiency, and performance, offering a more principled and effective paradigm for LLM alignment.Our codes are open-sourced via https://github.com/hlxtsyj/AMFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06931v1">Automated Formalization via Conceptual Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Interactive theorem provers (ITPs) require manual formalization, which is labor-intensive and demands expert knowledge. While automated formalization offers a potential solution, it faces two major challenges: model hallucination (e.g., undefined predicates, symbol misuse, and version incompatibility) and the semantic gap caused by ambiguous or missing premises in natural language descriptions. To address these issues, we propose CRAMF, a Concept-driven Retrieval-Augmented Mathematical Formalization framework. CRAMF enhances LLM-based autoformalization by retrieving formal definitions of core mathematical concepts, providing contextual grounding during code generation. However, applying retrieval-augmented generation (RAG) in this setting is non-trivial due to the lack of structured knowledge bases, the polymorphic nature of mathematical concepts, and the high precision required in formal retrieval. We introduce a framework for automatically constructing a concept-definition knowledge base from Mathlib4, the standard mathematical library for the Lean 4 theorem prover, indexing over 26,000 formal definitions and 1,000+ core mathematical concepts. To address conceptual polymorphism, we propose contextual query augmentation with domain- and application-level signals. In addition, we design a dual-channel hybrid retrieval strategy with reranking to ensure accurate and relevant definition retrieval. Experiments on miniF2F, ProofNet, and our newly proposed AdvancedMath benchmark show that CRAMF can be seamlessly integrated into LLM-based autoformalizers, yielding consistent improvements in translation accuracy, achieving up to 62.1% and an average of 29.9% relative improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06926v1">Integrating Rules and Semantics for LLM-Based C-to-Rust Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 Accepted in ICSME 25 Industry Track
    </div>
    <details class="paper-abstract">
      Automated translation of legacy C code into Rust aims to ensure memory safety while reducing the burden of manual migration. Early approaches in code translation rely on static rule-based methods, but they suffer from limited coverage due to dependence on predefined rule patterns. Recent works regard the task as a sequence-to-sequence problem by leveraging large language models (LLMs). Although these LLM-based methods are capable of reducing unsafe code blocks, the translated code often exhibits issues in following Rust rules and maintaining semantic consistency. On one hand, existing methods adopt a direct prompting strategy to translate the C code, which struggles to accommodate the syntactic rules between C and Rust. On the other hand, this strategy makes it difficult for LLMs to accurately capture the semantics of complex code. To address these challenges, we propose IRENE, an LLM-based framework that Integrates RulEs aNd sEmantics to enhance translation. IRENE consists of three modules: 1) a rule-augmented retrieval module that selects relevant translation examples based on rules generated from a static analyzer developed by us, thereby improving the handling of Rust rules; 2) a structured summarization module that produces a structured summary for guiding LLMs to enhance the semantic understanding of C code; 3) an error-driven translation module that leverages compiler diagnostics to iteratively refine translations. We evaluate IRENE on two datasets (xCodeEval, a public dataset, and HW-Bench, an industrial dataset provided by Huawei) and eight LLMs, focusing on translation accuracy and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06917v1">CROP: Integrating Topological and Spatial Structures via Cross-View Prefixes for Molecular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 Accepted to ACMMM 2025
    </div>
    <details class="paper-abstract">
      Recent advances in molecular science have been propelled significantly by large language models (LLMs). However, their effectiveness is limited when relying solely on molecular sequences, which fail to capture the complex structures of molecules. Beyond sequence representation, molecules exhibit two complementary structural views: the first focuses on the topological relationships between atoms, as exemplified by the graph view; and the second emphasizes the spatial configuration of molecules, as represented by the image view. The two types of views provide unique insights into molecular structures. To leverage these views collaboratively, we propose the CROss-view Prefixes (CROP) to enhance LLMs' molecular understanding through efficient multi-view integration. CROP possesses two advantages: (i) efficiency: by jointly resampling multiple structural views into fixed-length prefixes, it avoids excessive consumption of the LLM's limited context length and allows easy expansion to more views; (ii) effectiveness: by utilizing the LLM's self-encoded molecular sequences to guide the resampling process, it boosts the quality of the generated prefixes. Specifically, our framework features a carefully designed SMILES Guided Resampler for view resampling, and a Structural Embedding Gate for converting the resulting embeddings into LLM's prefixes. Extensive experiments demonstrate the superiority of CROP in tasks including molecule captioning, IUPAC name prediction and molecule property prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06913v1">Model-Agnostic Sentiment Distribution Stability Analysis for Robust LLM-Generated Texts Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has resulted in increasingly sophisticated AI-generated content, posing significant challenges in distinguishing LLM-generated text from human-written language. Existing detection methods, primarily based on lexical heuristics or fine-tuned classifiers, often suffer from limited generalizability and are vulnerable to paraphrasing, adversarial perturbations, and cross-domain shifts. In this work, we propose SentiDetect, a model-agnostic framework for detecting LLM-generated text by analyzing the divergence in sentiment distribution stability. Our method is motivated by the empirical observation that LLM outputs tend to exhibit emotionally consistent patterns, whereas human-written texts display greater emotional variability. To capture this phenomenon, we define two complementary metrics: sentiment distribution consistency and sentiment distribution preservation, which quantify stability under sentiment-altering and semantic-preserving transformations. We evaluate SentiDetect on five diverse datasets and a range of advanced LLMs,including Gemini-1.5-Pro, Claude-3, GPT-4-0613, and LLaMa-3.3. Experimental results demonstrate its superiority over state-of-the-art baselines, with over 16% and 11% F1 score improvements on Gemini-1.5-Pro and GPT-4-0613, respectively. Moreover, SentiDetect also shows greater robustness to paraphrasing, adversarial attacks, and text length variations, outperforming existing detectors in challenging scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06846v1">Highlight All the Phrases: Enhancing LLM Transparency through Visual Factuality Indicators</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 16 pages, 8 figures, To be published in Proceedings of the 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are susceptible to generating inaccurate or false information, often referred to as "hallucinations" or "confabulations." While several technical advancements have been made to detect hallucinated content by assessing the factuality of the model's responses, there is still limited research on how to effectively communicate this information to users. To address this gap, we conducted two scenario-based experiments with a total of 208 participants to systematically compare the effects of various design strategies for communicating factuality scores by assessing participants' ratings of trust, ease in validating response accuracy, and preference. Our findings reveal that participants preferred and trusted a design in which all phrases within a response were color-coded based on factuality scores. Participants also found it easier to validate accuracy of the response in this style compared to a baseline with no style applied. Our study offers practical design guidelines for LLM application developers and designers, aimed at calibrating user trust, aligning with user preferences, and enhancing users' ability to scrutinize LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19997v2">Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing education, with LLM-based agents playing a key role in simulating student behavior. A major challenge in student simulation is modeling the diverse learning patterns of students at various cognitive levels. However, current LLMs, typically trained as ``helpful assistants'', target at generating perfect responses. As a result, they struggle to simulate students with diverse cognitive abilities, as they often produce overly advanced answers, missing the natural imperfections that characterize student learning and resulting in unrealistic simulations. To address this issue, we propose a training-free framework for student simulation. We begin by constructing a cognitive prototype for each student using a knowledge graph, which captures their understanding of concepts from past learning records. This prototype is then mapped to new tasks to predict student performance. Next, we simulate student solutions based on these predictions and iteratively refine them using a beam search method to better replicate realistic mistakes. To validate our approach, we construct the \texttt{Student\_100} dataset, consisting of $100$ students working on Python programming and $5,000$ learning records. Experimental results show that our method consistently outperforms baseline models, achieving $100\%$ improvement in simulation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05660v3">Not Like Us, Hunty: Measuring Perceptions and Behavioral Effects of Minoritized Anthropomorphic Cues in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 accepted to FAccT 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly adapt and personalize to diverse sets of users, there is an increased risk of systems appropriating sociolects, i.e., language styles or dialects that are associated with specific minoritized lived experiences (e.g., African American English, Queer slang). In this work, we examine whether sociolect usage by an LLM agent affects user reliance on its outputs and user perception (satisfaction, frustration, trust, and social presence). We designed and conducted user studies where 498 African American English (AAE) speakers and 487 Queer slang speakers performed a set of question-answering tasks with LLM-based suggestions in either standard American English (SAE) or their self-identified sociolect. Our findings showed that sociolect usage by LLMs influenced both reliance and perceptions, though in some surprising ways. Results suggest that both AAE and Queer slang speakers relied more on the SAE agent, and had more positive perceptions of the SAE agent. Yet, only Queer slang speakers felt more social presence from the Queer slang agent over the SAE one, whereas only AAE speakers preferred and trusted the SAE agent over the AAE one. These findings emphasize the need to test for behavioral outcomes rather than simply assume that personalization would lead to a better and safer reliance outcome. They also highlight the nuanced dynamics of minoritized language in machine interactions, underscoring the need for LLMs to be carefully designed to respect cultural and linguistic boundaries while fostering genuine user engagement and trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19952v2">CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06799v1">LSDTs: LLM-Augmented Semantic Digital Twins for Adaptive Knowledge-Intensive Infrastructure Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Digital Twins (DTs) offer powerful tools for managing complex infrastructure systems, but their effectiveness is often limited by challenges in integrating unstructured knowledge. Recent advances in Large Language Models (LLMs) bring new potential to address this gap, with strong abilities in extracting and organizing diverse textual information. We therefore propose LSDTs (LLM-Augmented Semantic Digital Twins), a framework that helps LLMs extract planning knowledge from unstructured documents like environmental regulations and technical guidelines, and organize it into a formal ontology. This ontology forms a semantic layer that powers a digital twin-a virtual model of the physical system-allowing it to simulate realistic, regulation-aware planning scenarios. We evaluate LSDTs through a case study of offshore wind farm planning in Maryland, including its application during Hurricane Sandy. Results demonstrate that LSDTs support interpretable, regulation-aware layout optimization, enable high-fidelity simulation, and enhance adaptability in infrastructure planning. This work shows the potential of combining generative AI with digital twins to support complex, knowledge-driven planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06765v1">Fed MobiLLM: Efficient Federated LLM Fine-Tuning over Heterogeneous Mobile Devices via Server Assisted Side-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Collaboratively fine-tuning (FT) large language models (LLMs) over heterogeneous mobile devices fosters immense potential applications of personalized intelligence. However, such a vision faces critical system challenges. Conventional federated LLM FT approaches place prohibitive computational and memory burdens on mobile hardware, and their synchronous model aggregation protocols stall for slower devices. In this paper, we propose Fed MobiLLM, a novel design to facilitate efficient federated LLM FT across mobile devices with diverse computing/communication speeds and local model architectures. In particular, Fed MobiLLM implements a pioneering server-assisted federated side-tuning paradigm. Briefly, mobile devices perform lightweight forward propagation computations on local data using their frozen pre-scaled backbone LLMs, and then upload selected intermediate activations. The server trains a shared side-network independently, eliminating client-side backpropagation and enabling asynchronous updates. To bridge model heterogeneity across different devices, we introduce an adaptive layer-wise feature alignment method, which ensures consistent representations for collaboratively tuning a shared side network. Extensive experimental results demonstrate that Fed MobiLLM can maintain robust fine-tuning performance while achieving extremely low on-device memory, with at least 95.2% reduction in computation overhead, 93.2% reduction in communication costs and 5.1x faster convergence compared to existing methods, validating its efficacy for practical LLM adaptation over heterogeneous mobile devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06760v1">Understanding Privacy Norms Around LLM-Based Chatbots: A Contextual Integrity Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      LLM-driven chatbots like ChatGPT have created large volumes of conversational data, but little is known about how user privacy expectations are evolving with this technology. We conduct a survey experiment with 300 US ChatGPT users to understand emerging privacy norms for sharing chatbot data. Our findings reveal a stark disconnect between user concerns and behavior: 82% of respondents rated chatbot conversations as sensitive or highly sensitive - more than email or social media posts - but nearly half reported discussing health topics and over one-third discussed personal finances with ChatGPT. Participants expressed strong privacy concerns (t(299) = 8.5, p < .01) and doubted their conversations would remain private (t(299) = -6.9, p < .01). Despite this, respondents uniformly rejected sharing personal data (search history, emails, device access) for improved services, even in exchange for premium features worth $200. To identify which factors influence appropriate chatbot data sharing, we presented participants with factorial vignettes manipulating seven contextual factors. Linear mixed models revealed that only the transmission factors such as informed consent, data anonymization, or the removal of personally identifiable information, significantly affected perceptions of appropriateness and concern for data access. Surprisingly, contextual factors including the recipient of the data (hospital vs. tech company), purpose (research vs. advertising), type of content, and geographic location did not show significant effects. Our results suggest that users apply consistent baseline privacy expectations to chatbot data, prioritizing procedural safeguards over recipient trustworthiness. This has important implications for emerging agentic AI systems that assume user willingness to integrate personal data across platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01216v2">PAE MobiLLM: Privacy-Aware and Efficient LLM Fine-Tuning on the Mobile Device via Additive Side-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      There is a huge gap between numerous intriguing applications fostered by on-device large language model (LLM) fine-tuning (FT) from fresh mobile data and the limited resources of a mobile device. While existing server-assisted methods (e.g., split learning or side-tuning) may enable LLM FT on the local mobile device, they suffer from heavy communication burdens of activation transmissions, and may disclose data and labels to the server. To address those issues, we develop PAE MobiLLM, a a privacy-aware and efficient LLM FT method which can be deployed on the mobile device via server-assisted additive side-tuning. To further accelerate FT convergence and improve computing efficiency, PAE MobiLLM integrates activation caching on the server side, which allows the server to reuse historical activations and saves the mobile device from repeatedly computing forward passes for the recurring data samples. Besides, to reduce communication cost, PAE MobiLLM develops an activation shortcut that transmits only the token involved in the loss calculation instead of full activation matrices to guide the side network tuning. Last but not least, PAE MobiLLM introduces the additive adapter side-network design which makes the server train the adapter modules based on device-defined prediction differences rather than raw ground-truth labels. In this way, the server can only assist device-defined side-network computing, and learn nothing about data and labels. Extensive experimental results demonstrate PAE MobiLLM's superiority.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08322v1">Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code</a></div>
    <div class="paper-meta">
      📅 2025-08-09
      | 💬 15 pages, 5 figures, research paper on multi-agent LLM systems for code generation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in automating code generation and software engineering tasks, yet they often struggle with complex, multi-file projects due to context limitations and knowledge gaps. We propose a novel context engineering workflow that combines multiple AI components: an Intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a Claude Code multi-agent system for code generation and validation. Our integrated approach leverages intent clarification, retrieval-augmented generation, and specialized sub-agents orchestrated via Claude's agent framework. We demonstrate that this method significantly improves the accuracy and reliability of code assistants in real-world repositories, yielding higher single-shot success rates and better adherence to project context than baseline single-agent approaches. Qualitative results on a large Next.js codebase show the multi-agent system effectively plans, edits, and tests complex features with minimal human intervention. We compare our system with recent frameworks like CodePlan, MASAI, and HyperAgent, highlighting how targeted context injection and agent role decomposition lead to state-of-the-art performance. Finally, we discuss the implications for deploying LLM-based coding assistants in production, along with lessons learned on context management and future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06133v1">LLM Serving Optimization with Variable Prefill and Decode Lengths</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      We study the problem of serving LLM (Large Language Model) requests where each request has heterogeneous prefill and decode lengths. In LLM serving, the prefill length corresponds to the input prompt length, which determines the initial memory usage in the KV cache. The decode length refers to the number of output tokens generated sequentially, with each additional token increasing the KV cache memory usage by one unit. Given a set of n requests, our goal is to schedule and process them to minimize the total completion time. We show that this problem is NP-hard due to the interplay of batching, placement constraints, precedence relationships, and linearly increasing memory usage. We then analyze commonly used scheduling strategies in practice, such as First-Come-First-Serve (FCFS) and Shortest-First (SF), and prove that their competitive ratios scale up sublinearly with the memory limit-a significant drawback in real-world settings where memory demand is large. To address this, we propose a novel algorithm based on a new selection metric that efficiently forms batches over time. We prove that this algorithm achieves a constant competitive ratio. Finally, we develop and evaluate a few algorithm variants inspired by this approach, including dynamic programming variants, local search methods, and an LP-based scheduler, demonstrating through comprehensive simulations that they outperform standard baselines while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06111v1">SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 7 pages and appendices
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities and risks of foundation models is paramount, yet current methods demand extensive domain expertise, hindering their scalability as these models rapidly evolve. We introduce SKATE: a novel evaluation framework in which large language models (LLMs) compete by generating and solving verifiable tasks for one another. Our core insight is to treat evaluation as a game: models act as both task-setters and solvers, incentivized to create questions which highlight their own strengths while exposing others' weaknesses. SKATE offers several key advantages, balancing scalability, open-endedness, and objectivity. It is fully automated, data-free, and scalable, requiring no human input or domain expertise. By using verifiable tasks rather than LLM judges, scoring is objective. Unlike domain-limited programmatically-generated benchmarks (e.g. chess-playing or spatial reasoning), having LLMs creatively pose challenges enables open-ended and scalable evaluation. As a proof of concept, we introduce LLM-set code-output-prediction (COP) challenges as a verifiable and extensible framework in which to test our approach. Using a TrueSkill-based ranking system, we evaluate six frontier LLMs and find that: (1) weaker models can reliably differentiate and score stronger ones, (2) LLM-based systems are capable of self-preferencing behavior, generating questions that align with their own capabilities, and (3) SKATE automatically surfaces fine-grained capability differences between models. Our findings are an important step towards general, scalable evaluation frameworks which can keep pace with LLM progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06094v1">ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Project page: https://conlangcrafter.github.io
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, large-scale foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages -- phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' meta-linguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We evaluate ConlangCrafter on metrics measuring coherence and typological diversity, demonstrating its ability to produce coherent and varied conlangs without human linguistic expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17631v2">Time-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23145v2">CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Inductive program synthesis, or programming by example, requires synthesizing functions from input-output examples that generalize to unseen inputs. While large language model agents have shown promise in programming tasks guided by natural language, their ability to perform inductive program synthesis is underexplored. Existing evaluation protocols rely on static sets of examples and held-out tests, offering no feedback when synthesized functions are incorrect and failing to reflect real-world scenarios such as reverse engineering. We propose CodeARC, the Code Abstraction and Reasoning Challenge, a new evaluation framework where agents interact with a hidden target function by querying it with new inputs, synthesizing candidate functions, and iteratively refining their solutions using a differential testing oracle. This interactive setting encourages agents to perform function calls and self-correction based on feedback. We construct the first large-scale benchmark for general-purpose inductive program synthesis, featuring 1114 functions. Among 18 models evaluated, o3-mini performs best with a success rate of 52.7%, highlighting the difficulty of this task. Fine-tuning LLaMA-3.1-8B-Instruct on curated synthesis traces yields up to a 31% relative performance gain. CodeARC provides a more realistic and challenging testbed for evaluating LLM-based program synthesis and inductive reasoning. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/CodeARC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18602v2">LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized algorithm development, yet their application in symbolic regression, where algorithms automatically discover symbolic expressions from data, remains constrained and is typically designed manually by human experts. In this paper, we propose a meta learning framework that enables LLMs to automatically design selection operators for evolutionary symbolic regression algorithms. We first identify two key limitations in existing LLM-based algorithm evolution techniques: a lack of semantic guidance and code bloat. The absence of semantic awareness can lead to ineffective exchange of useful code components, and bloat results in unnecessarily complex components, both of which can reduce the interpretability of the designed algorithm or hinder evolutionary learning progress. To address these issues, we enhance the LLM-based evolution framework for meta symbolic regression with two key innovations: a complementary, semantics-aware selection operator and bloat control. Additionally, we embed domain knowledge into the prompt, enabling the LLM to generate more effective and contextually relevant selection operators. Our experimental results on symbolic regression benchmarks show that LLMs can devise selection operators that outperform nine expert-designed baselines, achieving state-of-the-art performance. Moreover, the evolved operator can further improve the state-of-the-art symbolic regression algorithm, achieving the best performance among 26 symbolic regression and machine learning algorithms across 116 regression datasets. This demonstrates that LLMs can exceed expert-level algorithm design for symbolic regression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06060v1">LLMs for Resource Allocation: A Participatory Budgeting Approach to Inferring Preferences</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Published in the Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to handle complex decision-making tasks, yet their ability to perform structured resource allocation remains underexplored. Evaluating their reasoning is also difficult due to data contamination and the static nature of existing benchmarks. We present a dual-purpose framework leveraging Participatory Budgeting (PB) both as (i) a practical setting for LLM-based resource allocation and (ii) an adaptive benchmark for evaluating their reasoning capabilities. We task LLMs with selecting project subsets under feasibility (e.g., budget) constraints via three prompting strategies: greedy selection, direct optimization, and a hill-climbing-inspired refinement. We benchmark LLMs' allocations against a utility-maximizing oracle. Interestingly, we also test whether LLMs can infer structured preferences from natural-language voter input or metadata, without explicit votes. By comparing allocations based on inferred preferences to those from ground-truth votes, we evaluate LLMs' ability to extract preferences from open-ended input. Our results underscore the role of prompt design and show that LLMs hold promise for mechanism design with unstructured inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v4">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06047v1">ArchXBench: A Complex Digital Systems Benchmark Suite for LLM Driven RTL Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Published in 7th ACM/IEEE International Symposium on Machine Learning for CAD
    </div>
    <details class="paper-abstract">
      Modern SoC datapaths include deeply pipelined, domain-specific accelerators, but their RTL implementation and verification are still mostly done by hand. While large language models (LLMs) exhibit advanced code-generation abilities for programming languages like Python, their application to Verilog-like RTL remains in its nascent stage. This is reflected in the simple arithmetic and control circuits currently used to evaluate generative capabilities in existing benchmarks. In this paper, we introduce ArchXBench, a six-level benchmark suite that encompasses complex arithmetic circuits and other advanced digital subsystems drawn from domains such as cryptography, image processing, machine learning, and signal processing. Architecturally, some of these designs are purely combinational, others are multi-cycle or pipelined, and many require hierarchical composition of modules. For each benchmark, we provide a problem description, design specification, and testbench, enabling rapid research in the area of LLM-driven agentic approaches for complex digital systems design. Using zero-shot prompting with Claude Sonnet 4, GPT 4.1, o4-mini-high, and DeepSeek R1 under a pass@5 criterion, we observed that o4-mini-high successfully solves the largest number of benchmarks, 16 out of 30, spanning Levels 1, 2, and 3. From Level 4 onward, however, all models consistently fail, highlighting a clear gap in the capabilities of current state-of-the-art LLMs and prompting/agentic approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08775v3">Layers at Similar Depths Generate Similar Activations Across LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05298v2">GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 17 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      We present GhostShell, a novel approach that leverages Large Language Models (LLMs) to enable streaming and concurrent behavioral programming for embodied systems. In contrast to conventional methods that rely on pre-scheduled action sequences or behavior trees, GhostShell drives embodied systems to act on-the-fly by issuing function calls incrementally as tokens are streamed from the LLM. GhostShell features a streaming XML function token parser, a dynamic function interface mapper, and a multi-channel scheduler that orchestrates intra-channel synchronous and inter-channel asynchronous function calls, thereby coordinating serial-parallel embodied actions across multiple robotic components under LLM guidance. We evaluate GhostShell on our robotic prototype COCO through comprehensive grounded experiments across 34 real-world interaction tasks and multiple LLM backends. The results demonstrate that our approach achieves a state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4-Sonnet, and up to 66X faster response times compared to native LLM function calling APIs. GhostShell also proves effective in long-horizon multimodal tasks, exhibiting strong robustness and generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06004v1">When a Paper Has 1000 Authors: Rethinking Citation Metrics in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Author-level citation metrics provide a practical, interpretable, and scalable signal of scholarly influence in a complex research ecosystem. It has been widely used as a proxy in hiring decisions. However, the past five years have seen the rapid emergence of large-scale publications in the field of large language models and foundation models, with papers featuring hundreds to thousands of co-authors and receiving tens of thousands of citations within months. For example, Gemini has 1361 authors and has been cited around 4600 times in 19 months. In such cases, traditional metrics, such as total citation count and the $h$-index, fail to meaningfully distinguish individual contributions. Therefore, we propose the following research question: How can one identify standout researchers among thousands of co-authors in large-scale LLM papers? This question is particularly important in scenarios such as academic hiring and funding decisions. In this paper, we introduce a novel citation metric designed to address this challenge by balancing contributions across large-scale and small-scale publications. We propose the SBCI index, analyze its theoretical properties, and evaluate its behavior on synthetic publication datasets. Our results demonstrate that the proposed metric provides a more robust and discriminative assessment of individual scholarly impact in the era of large-scale collaborations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06000v1">Hand by Hand: LLM Driving EMS Assistant for Operational Skill Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      Operational skill learning, inherently physical and reliant on hands-on practice and kinesthetic feedback, has yet to be effectively replicated in large language model (LLM)-supported training. Current LLM training assistants primarily generate customized textual feedback, neglecting the crucial kinesthetic modality. This gap derives from the textual and uncertain nature of LLMs, compounded by concerns on user acceptance of LLM driven body control. To bridge this gap and realize the potential of collaborative human-LLM action, this work explores human experience of LLM driven kinesthetic assistance. Specifically, we introduced an "Align-Analyze-Adjust" strategy and developed FlightAxis, a tool that integrates LLM with Electrical Muscle Stimulation (EMS) for flight skill acquisition, a representative operational skill domain. FlightAxis learns flight skills from manuals and guides forearm movements during simulated flight tasks. Our results demonstrate high user acceptance of LLM-mediated body control and significantly reduced task completion times. Crucially, trainees reported that this kinesthetic assistance enhanced their awareness of operation flaws and fostered increased engagement in the training process, rather than relieving perceived load. This work demonstrated the potential of kinesthetic LLM training in operational skill acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05995v1">Optimizing Prompt Sequences using Monte Carlo Tree Search for LLM-Based Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in code generation and structured reasoning; however, their performance often degrades on complex tasks that require consistent multi-step planning. Recent work has explored combining LLMs with Monte Carlo Tree Search (MCTS), yet existing approaches primarily focus on generating heuristic-based code for optimization or target simpler tasks where correctness alone is sufficient. In this work, we propose MCTS-OPS, a novel neural-symbolic framework that formulates prompt selection as a sequential decision process guided by MCTS. Our method explores and refines multi-step prompt sequences for the goal of improving code generation quality and enhancing the problem-solving capabilities of LLMs in general optimization. Experiments on network optimization show significant improvement over the baselines, both in the success rate of executing the generated code and in the optimization results with the specified objective and constraints (2$\sim$4$\times$ higher reward and 3$\times$ lower standard deviation). Moreover, it improves the chance of attaining the optimal solution by about 10\% of cases, compared to baseline methods in hard problems. These results highlight the promise of combining symbolic planning with LLMs for robust, high-quality code generation in complex domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05028v2">Evaluation of LLMs in AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 27 pages, 32 figures
    </div>
    <details class="paper-abstract">
      AMR (Abstract Meaning Representation) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05954v1">Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Project Page: https://bifrost-1.github.io
    </div>
    <details class="paper-abstract">
      There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05953v1">SCALEFeedback: A Large-Scale Dataset of Synthetic Computer Science Assignments for LLM-generated Educational Feedback Research</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Using LLMs to give educational feedback to students for their assignments has attracted much attention in the AI in Education field. Yet, there is currently no large-scale open-source dataset of student assignments that includes detailed assignment descriptions, rubrics, and student submissions across various courses. As a result, research on generalisable methodology for automatic generation of effective and responsible educational feedback remains limited. In the current study, we constructed a large-scale dataset of Synthetic Computer science Assignments for LLM-generated Educational Feedback research (SCALEFeedback). We proposed a Sophisticated Assignment Mimicry (SAM) framework to generate the synthetic dataset by one-to-one LLM-based imitation from real assignment descriptions, student submissions to produce their synthetic versions. Our open-source dataset contains 10,000 synthetic student submissions spanning 155 assignments across 59 university-level computer science courses. Our synthetic submissions achieved BERTScore F1 0.84, PCC of 0.62 for assignment marks and 0.85 for length, compared to the corresponding real-world assignment dataset, while ensuring perfect protection of student private information. All these results of our SAM framework outperformed results of a naive mimicry method baseline. The LLM-generated feedback for our synthetic assignments demonstrated the same level of effectiveness compared to that of real-world assignment dataset. Our research showed that one-to-one LLM imitation is a promising method for generating open-source synthetic educational datasets that preserve the original dataset's semantic meaning and student data distribution, while protecting student privacy and institutional copyright. SCALEFeedback enhances our ability to develop LLM-based generalisable methods for offering high-quality, automated educational feedback in a scalable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05952v1">Dean of LLM Tutors: Exploring Comprehensive and Automated Evaluation of LLM-generated Educational Feedback via LLM Feedback Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The use of LLM tutors to provide automated educational feedback to students on student assignment submissions has received much attention in the AI in Education field. However, the stochastic nature and tendency for hallucinations in LLMs can undermine both quality of learning experience and adherence to ethical standards. To address this concern, we propose a method that uses LLM feedback evaluators (DeanLLMs) to automatically and comprehensively evaluate feedback generated by LLM tutor for submissions on university assignments before it is delivered to students. This allows low-quality feedback to be rejected and enables LLM tutors to improve the feedback they generated based on the evaluation results. We first proposed a comprehensive evaluation framework for LLM-generated educational feedback, comprising six dimensions for feedback content, seven for feedback effectiveness, and three for hallucination types. Next, we generated a virtual assignment submission dataset covering 85 university assignments from 43 computer science courses using eight commonly used commercial LLMs. We labelled and open-sourced the assignment dataset to support the fine-tuning and evaluation of LLM feedback evaluators. Our findings show that o3-pro demonstrated the best performance in zero-shot labelling of feedback while o4-mini demonstrated the best performance in few-shot labelling of feedback. Moreover, GPT-4.1 achieved human expert level performance after fine-tuning (Accuracy 79.8%, F1-score 79.4%; human average Accuracy 78.3%, F1-score 82.6%). Finally, we used our best-performance model to evaluate 2,000 assignment feedback instances generated by 10 common commercial LLMs, 200 each, to compare the quality of feedback generated by different LLMs. Our LLM feedback evaluator method advances our ability to automatically provide high-quality and reliable educational feedback to students.
    </details>
</div>
