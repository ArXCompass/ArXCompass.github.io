# llm - 2025_04

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
- Part 12

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01205v1">Epistemic Alignment: A Mediating Framework for User-LLM Knowledge Delivery</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLMs increasingly serve as tools for knowledge acquisition, yet users cannot effectively specify how they want information presented. When users request that LLMs "cite reputable sources," "express appropriate uncertainty," or "include multiple perspectives," they discover that current interfaces provide no structured way to articulate these preferences. The result is prompt sharing folklore: community-specific copied prompts passed through trust relationships rather than based on measured efficacy. We propose the Epistemic Alignment Framework, a set of ten challenges in knowledge transmission derived from the philosophical literature of epistemology, concerning issues such as evidence quality assessment and calibration of testimonial reliance. The framework serves as a structured intermediary between user needs and system capabilities, creating a common vocabulary to bridge the gap between what users want and what systems deliver. Through a thematic analysis of custom prompts and personalization strategies shared on online communities where these issues are actively discussed, we find users develop elaborate workarounds to address each of the challenges. We then apply our framework to two prominent model providers, OpenAI and Anthropic, through content analysis of their documented policies and product features. Our analysis shows that while these providers have partially addressed the challenges we identified, they fail to establish adequate mechanisms for specifying epistemic preferences, lack transparency about how preferences are implemented, and offer no verification tools to confirm whether preferences were followed. For AI developers, the Epistemic Alignment Framework offers concrete guidance for supporting diverse approaches to knowledge; for users, it works toward information delivery that aligns with their specific needs rather than defaulting to one-size-fits-all approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01145v1">MaLAware: Automating the Comprehension of Malicious Software Behaviours using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at MSR 2025
    </div>
    <details class="paper-abstract">
      Current malware (malicious software) analysis tools focus on detection and family classification but fail to provide clear and actionable narrative insights into the malignant activity of the malware. Therefore, there is a need for a tool that translates raw malware data into human-readable descriptions. Developing such a tool accelerates incident response, reduces malware analysts' cognitive load, and enables individuals having limited technical expertise to understand malicious software behaviour. With this objective, we present MaLAware, which automatically summarizes the full spectrum of malicious activity of malware executables. MaLAware processes Cuckoo Sandbox-generated reports using large language models (LLMs) to correlate malignant activities and generate concise summaries explaining malware behaviour. We evaluate the tool's performance on five open-source LLMs. The evaluation uses the human-written malware behaviour description dataset as ground truth. The model's performance is measured using 11 extensive performance metrics, which boosts the confidence of MaLAware's effectiveness. The current version of the tool, i.e., MaLAware, supports Qwen2.5-7B, Llama2-7B, Llama3.1-8B, Mistral-7B, and Falcon-7B, along with the quantization feature for resource-constrained environments. MaLAware lays a foundation for future research in malware behavior explanation, and its extensive evaluation demonstrates LLMs' ability to narrate malware behavior in an actionable and comprehensive manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05396v2">FairCoder: Evaluating Social Bias of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed in coding tasks, drawing increasing attention to the evaluation of the quality and safety of LLMs' outputs. However, research on bias in code generation remains limited. Existing studies typically identify bias by applying malicious prompts or reusing tasks and dataset originally designed for discriminative models. Given that prior datasets are not fully optimized for code-related tasks, there is a pressing need for benchmarks specifically designed for evaluating code models. In this study, we introduce FairCoder, a novel benchmark for evaluating social bias in code generation. FairCoder explores the bias issue following the pipeline in software development, from function implementation to unit test, with diverse real-world scenarios. Additionally, three metrics are designed to assess fairness performance on this benchmark. We conduct experiments on widely used LLMs and provide a comprehensive analysis of the results. The findings reveal that all tested LLMs exhibit social bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01127v1">Can LLMs Grasp Implicit Cultural Values? Benchmarking LLMs' Metacognitive Cultural Intelligence with CQ-Bench</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Cultural Intelligence (CQ) refers to the ability to understand unfamiliar cultural contexts-a crucial skill for large language models (LLMs) to effectively engage with globally diverse users. While existing research often focuses on explicitly stated cultural norms, such approaches fail to capture the subtle, implicit values that underlie real-world conversations. To address this gap, we introduce CQ-Bench, a benchmark specifically designed to assess LLMs' capability to infer implicit cultural values from natural conversational contexts. We generate a multi-character conversation-based stories dataset using values from the World Value Survey and GlobalOpinions datasets, with topics including ethical, religious, social, and political. Our dataset construction pipeline includes rigorous validation procedures-incorporation, consistency, and implicitness checks-using GPT-4o, with 98.2% human-model agreement in the final validation. Our benchmark consists of three tasks of increasing complexity: attitude detection, value selection, and value extraction. We find that while o1 and Deepseek-R1 models reach human-level performance in value selection (0.809 and 0.814), they still fall short in nuanced attitude detection, with F1 scores of 0.622 and 0.635, respectively. In the value extraction task, GPT-4o-mini and o3-mini score 0.602 and 0.598, highlighting the difficulty of open-ended cultural reasoning. Notably, fine-tuning smaller models (e.g., LLaMA-3.2-3B) on only 500 culturally rich examples improves performance by over 10%, even outperforming stronger baselines (o3-mini) in some cases. Using CQ-Bench, we provide insights into the current challenges in LLMs' CQ research and suggest practical pathways for enhancing LLMs' cross-cultural reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01094v1">Multilingual and Multi-Accent Jailbreaking of Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 21 pages, 6 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01005v1">When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00993v1">MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00858v1">Whispering Under the Eaves: Protecting User Privacy Against Commercial and LLM-powered Automatic Speech Recognition Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accept to USENIX Security 2025
    </div>
    <details class="paper-abstract">
      The widespread application of automatic speech recognition (ASR) supports large-scale voice surveillance, raising concerns about privacy among users. In this paper, we concentrate on using adversarial examples to mitigate unauthorized disclosure of speech privacy thwarted by potential eavesdroppers in speech communications. While audio adversarial examples have demonstrated the capability to mislead ASR models or evade ASR surveillance, they are typically constructed through time-intensive offline optimization, restricting their practicality in real-time voice communication. Recent work overcame this limitation by generating universal adversarial perturbations (UAPs) and enhancing their transferability for black-box scenarios. However, they introduced excessive noise that significantly degrades audio quality and affects human perception, thereby limiting their effectiveness in practical scenarios. To address this limitation and protect live users' speech against ASR systems, we propose a novel framework, AudioShield. Central to this framework is the concept of Transferable Universal Adversarial Perturbations in the Latent Space (LS-TUAP). By transferring the perturbations to the latent space, the audio quality is preserved to a large extent. Additionally, we propose target feature adaptation to enhance the transferability of UAPs by embedding target text features into the perturbations. Comprehensive evaluation on four commercial ASR APIs (Google, Amazon, iFlytek, and Alibaba), three voice assistants, two LLM-powered ASR and one NN-based ASR demonstrates the protection superiority of AudioShield over existing competitors, and both objective and subjective evaluations indicate that AudioShield significantly improves the audio quality. Moreover, AudioShield also shows high effectiveness in real-time end-to-end scenarios, and demonstrates strong resilience against adaptive countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00829v1">How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of Large Language Models (LLMs) with efficiency and scalability remains a fundamental challenge in artificial intelligence research. This paper presents a rigorous experimental investigation into how difficulty-aware staged reinforcement learning (RL) strategies can substantially improve LLM reasoning performance. Through systematic analysis, we demonstrate that strategically selecting training data according to well-defined difficulty levels markedly enhances RL optimization. Moreover, we introduce a staged training methodology, progressively exposing models to increasingly challenging tasks, further amplifying reasoning capabilities. Our findings reveal significant cross-domain benefits when simultaneously training models on mathematical reasoning and code generation tasks. Notably, our proposed approach enables a 1.5B parameter model to achieve an accuracy of 42.3\% on the AIME-2024 benchmark, 89.5\% on the MATH-500 benchmark. These results underscore the efficacy of our method in advancing the reasoning proficiency of LLMs. We will open-source our datasets on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00727v1">Personality-Driven Decision-Making in LLM-Based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 8 figures. To be included in Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)
    </div>
    <details class="paper-abstract">
      The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00711v1">GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The era of foundation models has revolutionized AI research, yet Graph Foundation Models (GFMs) remain constrained by the scarcity of large-scale graph corpora. Traditional graph data synthesis techniques primarily focus on simplistic structural operations, lacking the capacity to generate semantically rich nodes with meaningful textual attributes: a critical limitation for real-world applications. While large language models (LLMs) demonstrate exceptional text generation capabilities, their direct application to graph synthesis is impeded by context window limitations, hallucination phenomena, and structural consistency challenges. To address these issues, we introduce GraphMaster, the first multi-agent framework specifically designed for graph data synthesis in data-limited environments. GraphMaster orchestrates four specialized LLM agents (Manager, Perception, Enhancement, and Evaluation) that collaboratively optimize the synthesis process through iterative refinement, ensuring both semantic coherence and structural integrity. To rigorously evaluate our approach, we create new data-limited "Sub" variants of six standard graph benchmarks, specifically designed to test synthesis capabilities under realistic constraints. Additionally, we develop a novel interpretability assessment framework that combines human evaluation with a principled Grassmannian manifold-based analysis, providing both qualitative and quantitative measures of semantic coherence. Experimental results demonstrate that GraphMaster significantly outperforms traditional synthesis methods across multiple datasets, establishing a strong foundation for advancing GFMs in data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2408.13006v2">Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by Building Trust in LLMs and LLM Applications workshop at ICLR 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has been widely applied to evaluate and compare different LLM alignmnet approaches (e.g., RLHF and DPO). However, concerns regarding its reliability have emerged, due to LLM judges' biases and inconsistent decision-making. Previous research has developed evaluation frameworks to assess reliability of LLM judges and their alignment with human preferences. However, the employed evaluation metrics often lack adequate explainability and fail to address LLM internal inconsistency. Additionally, existing studies inadequately explore the impact of various prompt templates when applying LLM-as-a-Judge methods, leading to potentially inconsistent comparisons between different alignment algorithms. In this work, we systematically evaluate LLM-as-a-Judge on alignment tasks by defining more theoretically interpretable evaluation metrics and explicitly mitigating LLM internal inconsistency from reliability metrics. We develop an open-source framework to evaluate, compare, and visualize the reliability and alignment of LLM judges, which facilitates practitioners to choose LLM judges for alignment tasks. In the experiments, we examine effects of diverse prompt templates on LLM-judge reliability and also demonstrate our developed framework by comparing various LLM judges on two common alignment datasets (i.e., TL;DR Summarization and HH-RLHF-Helpfulness). Our results indicate a significant impact of prompt templates on LLM judge performance, as well as a mediocre alignment level between the tested LLM judges and human evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15487v2">Multi-LLM Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization.
    </details>
</div>
