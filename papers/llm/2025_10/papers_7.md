# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18096v1">A Benchmark Dataset And LLMs Comparison For NFR Classification With Explainable AI</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Non-Functional Requirements (NFRs) play a critical role in determining the overall quality and user satisfaction of software systems. Accurately identifying and classifying NFRs is essential to ensure that software meets performance, usability, and reliability expectations. However, manual identification of NFRs from documentation is time-consuming and prone to errors, necessitating automated solutions. Before implementing any automated solution, a robust and comprehensive dataset is essential. To build such a dataset, we collected NFRs from various Project Charters and Open Source Software Documentation. This enhanced the technical depth and usability of an already existing NFR dataset. We categorized NFRs into sub-classes and identified needs using widely used Large Language Models to facilitate automation. After classifying the NFRs, we compared the classification results of the selected LLMs: RoBERTa, CodeBERT, Gemma-2, Phi-3, Mistral-8B, and Llama-3.1-8B using various evaluation metrics, including precision, recall, F1-score, and lime scores. Among these models, Gemma-2 achieved the best results with a precision of 0.87, recall of 0.89, and F1-score of 0.88, alongside a lime hit score of 78 out of 80. Phi-3 closely followed with a precision of 0.85, recall of 0.87, F1-score of 0.86, and the highest lime hit score of 79. By improving the contextual foundation, this integration enhanced the model's comprehension of technical aspects and user requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18095v1">SMaRT: Select, Mix, and ReinvenT -- A Strategy Fusion Framework for LLM-Driven Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have redefined complex task automation with exceptional generalization capabilities. Despite these advancements, state-of-the-art methods rely on single-strategy prompting, missing the synergy of diverse reasoning approaches. No single strategy excels universally, highlighting the need for frameworks that fuse strategies to maximize performance and ensure robustness. We introduce the Select, Mix, and ReinvenT (SMaRT) framework, an innovative strategy fusion approach designed to overcome this constraint by creating balanced and efficient solutions through the seamless integration of diverse reasoning strategies. Unlike existing methods, which employ LLMs merely as evaluators, SMaRT uses them as intelligent integrators, unlocking the "best of all worlds" across tasks. Extensive empirical evaluations across benchmarks in reasoning, planning, and sequential decision-making highlight the robustness and adaptability of SMaRT. The framework consistently outperforms state-of-the-art baselines in solution quality, constraint adherence, and performance metrics. This work redefines LLM-driven decision-making by pioneering a new paradigm in cross-strategy calibration, unlocking superior outcomes for reasoning systems and advancing the boundaries of self-refining methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18081v1">Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00657v3">LLM Safety Alignment is Divergence Estimation in Disguise</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We present a theoretical framework showing that popular LLM alignment methods, including RLHF and its variants, can be understood as divergence estimators between aligned (safe or preferred) and unaligned (harmful or less preferred) distributions. This perspective explains the emergence of separation in the latent space between safe and harmful prompts after alignment. As an application of our general divergence framework, we propose KLDO, a novel KL divergence-based alignment method, and empirically validate its effectiveness. We further show that using compliance-refusal datasets, rather than standard preference-based datasets, leads to stronger separation and improved safety alignment. Finally, to quantify the separation effect, we propose a distance-based metric in the prompt representation space, which also acts as a statistically significant indicator for model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14207v2">Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18043v1">CompactPrompt: A Unified Pipeline for Prompt Data Compression in LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Workshop on LLMs and Generative AI for Finance at ACM ICAIF 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver powerful reasoning and generation capabilities but incur substantial run-time costs when operating in agentic workflows that chain together lengthy prompts and process rich data streams. We introduce CompactPrompt, an end-to-end pipeline that merges hard prompt compression with lightweight file-level data compression. CompactPrompt first prunes low-information tokens from prompts using self-information scoring and dependency-based phrase grouping. In parallel, it applies n-gram abbreviation to recurrent textual patterns in attached documents and uniform quantization to numerical columns, yielding compact yet semantically faithful representations. Integrated into standard LLM agents, CompactPrompt reduces total token usage and inference cost by up to 60% on benchmark dataset like TAT-QA and FinQA, while preserving output quality (Results in less than 5% accuracy drop for Claude-3.5-Sonnet, and GPT-4.1-Mini) CompactPrompt helps visualize real-time compression decisions and quantify cost-performance trade-offs, laying the groundwork for leaner generative AI pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15216v2">Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18032v1">OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 8 pages for main content
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable reasoning capabilities in mathematical and scientific tasks. To enhance complex reasoning, multi-agent systems have been proposed to harness the collective intelligence of LLM agents. However, existing collaboration structures are either predefined or rely on majority voting or round-table debates, which can suppress correct but less dominant agent contributions. Recent approaches model multi-agent systems as graph networks but optimize purely for agent performance, neglecting the quality of interactions. We hypothesize that effective agent communication is crucial for multi-agent reasoning and that debating quality plays a significant role. To address this, we propose $\ours$, a multi-agent verbal reinforcement learning algorithm that dynamically constructs and refines multi-agent collaboration structures. Our method defines action spaces and a feedback mechanism that evaluates communication robustness and coherence throughout the debate. The final decision is achieved through a majority vote over all the agents. We assess $\ours$ on various reasoning tasks, including mathematical reasoning, creative writing, scientific reasoning, and numerical sorting. Results demonstrate that our approach significantly outperforms single-agent prompting methods and state-of-the-art multi-agent frameworks on diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18019v1">Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18003v1">BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19398v2">Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large language model-based (LLM-based) agents have become common in settings that include non-cooperative parties. In such settings, agents' decision-making needs to conceal information from their adversaries, reveal information to their cooperators, and infer information to identify the other agents' characteristics. To investigate whether LLMs have these information control and decision-making capabilities, we make LLM agents play the language-based hidden-identity game, The Chameleon. In this game, a group of non-chameleon agents who do not know each other aim to identify the chameleon agent without revealing a secret. The game requires the aforementioned information control capabilities both as a chameleon and a non-chameleon. We begin with a theoretical analysis for a spectrum of strategies, from concealing to revealing, and provide bounds on the non-chameleons' winning probability. The empirical results with GPT, Gemini 2.5 Pro, Llama 3.1, and Qwen3 models show that while non-chameleon LLM agents identify the chameleon, they fail to conceal the secret from the chameleon, and their winning probability is far from the levels of even trivial strategies. Based on these empirical results and our theoretical analysis, we deduce that LLM-based agents may reveal excessive information to agents of unknown identities. Interestingly, we find that, when instructed to adopt an information-revealing level, this level is linearly encoded in the LLM's internal representations. While the instructions alone are often ineffective at making non-chameleon LLMs conceal, we show that steering the internal representations in this linear direction directly can reliably induce concealing behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17941v1">Believe It or Not: How Deeply do LLMs Believe Implanted Facts?</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Knowledge editing techniques promise to implant new factual knowledge into large language models (LLMs). But do LLMs really believe these facts? We develop a framework to measure belief depth and use it to evaluate the success of knowledge editing techniques. We operationalize belief depth as the extent to which implanted knowledge 1) generalizes to related contexts (e.g. Fermi estimates several logical steps removed), 2) is robust to self-scrutiny and direct challenge, and 3) is represented similarly to genuine knowledge (as measured by linear probes). Our evaluations show that simple prompting and mechanistic editing techniques fail to implant knowledge deeply. In contrast, Synthetic Document Finetuning (SDF) - where models are trained on LLM-generated documents consistent with a fact - often succeeds at implanting beliefs that behave similarly to genuine knowledge. However, SDF's success is not universal, as implanted beliefs that contradict basic world knowledge are brittle and representationally distinct from genuine knowledge. Overall, our work introduces measurable criteria for belief depth and enables the rigorous evaluation necessary for deploying knowledge editing in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17934v1">AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has shown some success in augmenting large language models (LLMs) with external knowledge. However, as a non-parametric knowledge integration paradigm for LLMs, RAG methods heavily rely on external retrieval modules and the retrieved textual context prior. Especially for very large scale knowledge augmentation, they would introduce substantial inference latency due to expensive searches and much longer relevant context. In this paper, we propose a parametric knowledge integration method, called \textbf{AtlasKV}, a scalable, effective, and general way to augment LLMs with billion-scale knowledge graphs (KGs) (e.g. 1B triples) using very little GPU memory cost (e.g. less than 20GB VRAM). In AtlasKV, we introduce KG2KV and HiKVP to integrate KG triples into LLMs at scale with sub-linear time and memory complexity. It maintains strong knowledge grounding and generalization performance using the LLMs' inherent attention mechanism, and requires no external retrievers, long context priors, or retraining when adapting to new knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17924v1">Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Published in the Journal of Data Mining & Digital Humanities (JDMDH), special issue NLP4DH
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive comparative analysis of Natural Language Processing (NLP) methods for automated toxicity detection in online gaming chats. Traditional machine learning models with embeddings, large language models (LLMs) with zero-shot and few-shot prompting, fine-tuned transformer models, and retrieval-augmented generation (RAG) approaches are evaluated. The evaluation framework assesses three critical dimensions: classification accuracy, processing speed, and computational costs. A hybrid moderation system architecture is proposed that optimizes human moderator workload through automated detection and incorporates continuous learning mechanisms. The experimental results demonstrate significant performance variations across methods, with fine-tuned DistilBERT achieving optimal accuracy-cost trade-offs. The findings provide empirical evidence for deploying cost-effective, efficient content moderation systems in dynamic online gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17921v1">CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in enhancing the reasoning ability of large language models (LLMs) have been remarkably successful. LLMs trained with reinforcement learning (RL) for reasoning demonstrate strong performance in challenging tasks such as mathematics and coding, even with relatively small model sizes. However, despite these improvements in task accuracy, the assessment of creativity in LLM generations has been largely overlooked in reasoning tasks, in contrast to writing tasks. The lack of research on creativity assessment in reasoning primarily stems from two challenges: (1) the difficulty of defining the range of creativity, and (2) the necessity of human evaluation in the assessment process. To address these challenges, we propose CLAWS, a method that defines and classifies mathematical solutions into typical, creative, and hallucinated categories without human evaluation, by leveraging attention weights across prompt sections and output. CLAWS outperforms five existing white-box detection methods (Perplexity, Logit Entropy, Window Entropy, Hidden Score, and Attention Score) on five 7-8B math RL models (DeepSeek, Qwen, Mathstral, OpenMath2, and Oreal). We validate CLAWS on 4545 math problems collected from 181 math contests (AJHSME, AMC, AIME).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17918v1">JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The hallucination and credibility concerns of large language models (LLMs) are global challenges that the industry is collectively addressing. Recently, a significant amount of advances have been made on post-training and inference techniques to mitigate these challenges. However, it is widely agreed that unsafe and hallucinations of LLMs intrinsically originate from pre-training, involving pre-training data and the next-token prediction learning mechanism. In this paper, we focus on enhancing pre-training data to improve the trustworthiness and safety of LLMs. Since the data is vast, it's almost impossible to entirely purge the data of factual errors, logical inconsistencies, or distributional biases. Moreover, the pre-training data lack grounding in real-world knowledge. Each piece of data is treated as a sequence of tokens rather than as a representation of a part of the world. To overcome these issues, we propose approaches to enhancing our pre-training data with its context in the world and increasing a substantial amount of data reflecting industrial scenarios. We argue that most source data are created by the authors for specific purposes in a certain spatial-temporal context. They have played a role in the real world. By incorporating related world context information, we aim to better anchor pre-training data within real-world scenarios, thereby reducing uncertainty in model training and enhancing the model's safety and trustworthiness. We refer to our Data with World Context as DWC. We continue pre-training an earlier checkpoint of JT-35B-Base with 1.5 trillion of DWC tokens. We introduce our post-training procedures to activate the potentials of DWC. Compared with the Qwen model of a similar scale, JT-Safe-35B achieves an average performance improvement of 1.79% on the Safety and Trustworthy evaluation benchmarks, while being pretrained with only 6.2 trillion tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17727v1">Enabling Fine-Grained Operating Points for Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Black-box Large Language Models (LLMs) provide practical and accessible alternatives to other machine learning methods, as they require minimal labeled data and machine learning expertise to develop solutions for various decision making problems. However, for applications that need operating with constraints on specific metrics (e.g., precision $\geq$ 95%), decision making with black-box LLMs remains unfavorable, due to their low numerical output cardinalities. This results in limited control over their operating points, preventing fine-grained adjustment of their decision making behavior. In this paper, we study using black-box LLMs as classifiers, focusing on efficiently improving their operational granularity without performance loss. Specifically, we first investigate the reasons behind their low-cardinality numerical outputs and show that they are biased towards generating rounded but informative verbalized probabilities. Then, we experiment with standard prompt engineering, uncertainty estimation and confidence elicitation techniques, and observe that they do not effectively improve operational granularity without sacrificing performance or increasing inference cost. Finally, we propose efficient approaches to significantly increase the number and diversity of available operating points. Our proposed approaches provide finer-grained operating points and achieve comparable to or better performance than the benchmark methods across 11 datasets and 3 LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17726v1">Rethinking Search: A Study of University Students' Perspectives on Using LLMs and Traditional Search Engines in Academic Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Acctepted at the EMNLP 2025 HCI+NLP Workshop
    </div>
    <details class="paper-abstract">
      With the increasing integration of Artificial Intelligence (AI) in academic problem solving, university students frequently alternate between traditional search engines like Google and large language models (LLMs) for information retrieval. This study explores students' perceptions of both tools, emphasizing usability, efficiency, and their integration into academic workflows. Employing a mixed-methods approach, we surveyed 109 students from diverse disciplines and conducted in-depth interviews with 12 participants. Quantitative analyses, including ANOVA and chi-square tests, were used to assess differences in efficiency, satisfaction, and tool preference. Qualitative insights revealed that students commonly switch between GPT and Google: using Google for credible, multi-source information and GPT for summarization, explanation, and drafting. While neither tool proved sufficient on its own, there was a strong demand for a hybrid solution. In response, we developed a prototype, a chatbot embedded within the search interface, that combines GPT's conversational capabilities with Google's reliability to enhance academic research and reduce cognitive load.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17725v1">AcademicEval: Live Long-Context LLM Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted by TMLR. Code is available at https://github.com/ulab-uiuc/AcademicEval
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved remarkable performance in long-context understanding. However, current long-context LLM benchmarks are limited by rigid context length, labor-intensive annotation, and the pressing challenge of label leakage issues during LLM training. Therefore, we propose \textsc{AcademicEval}, a live benchmark for evaluating LLMs over long-context generation tasks. \textsc{AcademicEval} adopts papers on arXiv to introduce several academic writing tasks with long-context inputs, \textit{i.e.}, \textsc{Title}, \textsc{Abstract}, \textsc{Introduction}, and \textsc{Related Work}, which cover a wide range of abstraction levels and require no manual labeling. Moreover, \textsc{AcademicEval} integrates high-quality and expert-curated few-shot demonstrations from a collected co-author graph to enable flexible context length. Especially, \textsc{AcademicEval} features an efficient live evaluation, ensuring no label leakage. We conduct a holistic evaluation on \textsc{AcademicEval}, and the results illustrate that LLMs perform poorly on tasks with hierarchical abstraction levels and tend to struggle with long few-shot demonstrations, highlighting the challenge of our benchmark. Through experimental analysis, we also reveal some insights for enhancing LLMs' long-context modeling capabilities. Code is available at https://github.com/ulab-uiuc/AcademicEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05605v5">Evolving LLMs' Self-Refinement Capability via Iterative Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Self-Refinement refers to a model's ability to revise its own responses to produce improved outputs. This capability can also serve as a fundamental mechanism for Self-Improvement, for example, by reconstructing datasets with refined results to enhance intrinsic model performance. However, our comprehensive experiments reveal that large language models (LLMs) show no clear evidence of inherent Self-Refinement and may even experience response quality degradation after Self-Refinement. To address this issue, we propose EVOLVE, a simple and effective framework for eliciting and tracking the evolution of Self-Refinement through iterative training. We first explore optimization methods during training to activate the model's Self-Refinement capability. Then, at inference, we investigate various generation strategies to further enhance and utilize Self-Refinement while supplying the necessary data for training. Through synergistic optimization of training and inference stages, we continually evolve the model's Self-Refinement ability, enabling it to better refine its own responses. Moreover, we demonstrate the potential of leveraging Self-Refinement to achieve broader Self-Improvement of intrinsic model abilities. Experiments show that the evolved Self-Refinement ability enables the Llama-3.1-8B base model to surpass GPT-4o, achieving 62.3% length-controlled and 63.3% raw win rates on AlpacaEval 2, and 50.3% on Arena-Hard. It also generalizes effectively to out-of-domain reasoning tasks, improving performance on mathematical reasoning benchmarks such as GSM8K and MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17698v1">Towards Mining Effective Pedagogical Strategies from Learner-LLM Educational Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Dialogue plays a crucial role in educational settings, yet existing evaluation methods for educational applications of large language models (LLMs) primarily focus on technical performance or learning outcomes, often neglecting attention to learner-LLM interactions. To narrow this gap, this AIED Doctoral Consortium paper presents an ongoing study employing a dialogue analysis approach to identify effective pedagogical strategies from learner-LLM dialogues. The proposed approach involves dialogue data collection, dialogue act (DA) annotation, DA pattern mining, and predictive model building. Early insights are outlined as an initial step toward future research. The work underscores the need to evaluate LLM-based educational applications by focusing on dialogue dynamics and pedagogical strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12943v2">The Curious Case of Curiosity across Human Cultures and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Preprint (Paper under review)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have expanded their role in human interaction, yet curiosity -- a central driver of inquiry -- remains underexplored in these systems, particularly across cultural contexts. In this work, we investigate cultural variation in curiosity using Yahoo! Answers, a real-world multi-country dataset spanning diverse topics. We introduce CUEST (CUriosity Evaluation across SocieTies), an evaluation framework that measures human-model alignment in curiosity through linguistic (style), topic preference (content) analysis and grounding insights in social science constructs. Across open- and closed-source models, we find that LLMs flatten cross-cultural diversity, aligning more closely with how curiosity is expressed in Western countries. We then explore fine-tuning strategies to induce curiosity in LLMs, narrowing the human-model alignment gap by up to 50%. Finally, we demonstrate the practical value of curiosity for LLM adaptability across cultures, showing its importance for future NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17638v1">LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 https://www.prophetarena.co/
    </div>
    <details class="paper-abstract">
      Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12814v2">PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Pre-MIT Press publication version, has been accepted by TACL
    </div>
    <details class="paper-abstract">
      Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17603v1">ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025 Poster
    </div>
    <details class="paper-abstract">
      3D generation from natural language offers significant potential to reduce expert manual modeling efforts and enhance accessibility to 3D assets. However, existing methods often yield unstructured meshes and exhibit poor interactivity, making them impractical for artistic workflows. To address these limitations, we represent 3D assets as shape programs and introduce ShapeCraft, a novel multi-agent framework for text-to-3D generation. At its core, we propose a Graph-based Procedural Shape (GPS) representation that decomposes complex natural language into a structured graph of sub-tasks, thereby facilitating accurate LLM comprehension and interpretation of spatial relationships and semantic shape details. Specifically, LLM agents hierarchically parse user input to initialize GPS, then iteratively refine procedural modeling and painting to produce structured, textured, and interactive 3D assets. Qualitative and quantitative experiments demonstrate ShapeCraft's superior performance in generating geometrically accurate and semantically rich 3D assets compared to existing LLM-based agents. We further show the versatility of ShapeCraft through examples of animated and user-customized editing, highlighting its potential for broader interactive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00432v2">Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17576v1">Intent-Driven LLM Ensemble Planning for Flexible Multi-Robot Disassembly: Demonstration on EV Batteries</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 This work is funded by the project called "Research and Development of a Highly Automated and Safe Streamlined Process for Increasing Lithium-ion Battery Repurposing and Recycling" (REBELION) under Grant 101104241, and partially supported by the Ministry of National Education, Republic of Turkey. Submitted to Frontiers for Review
    </div>
    <details class="paper-abstract">
      This paper addresses the problem of planning complex manipulation tasks, in which multiple robots with different end-effectors and capabilities, informed by computer vision, must plan and execute concatenated sequences of actions on a variety of objects that can appear in arbitrary positions and configurations in unstructured scenes. We propose an intent-driven planning pipeline which can robustly construct such action sequences with varying degrees of supervisory input from a human using simple language instructions. The pipeline integrates: (i) perception-to-text scene encoding, (ii) an ensemble of large language models (LLMs) that generate candidate removal sequences based on the operator's intent, (iii) an LLM-based verifier that enforces formatting and precedence constraints, and (iv) a deterministic consistency filter that rejects hallucinated objects. The pipeline is evaluated on an example task in which two robot arms work collaboratively to dismantle an Electric Vehicle battery for recycling applications. A variety of components must be grasped and removed in specific sequences, determined by human instructions and/or by task-order feasibility decisions made by the autonomous system. On 200 real scenes with 600 operator prompts across five component classes, we used metrics of full-sequence correctness and next-task correctness to evaluate and compare five LLM-based planners (including ablation analyses of pipeline components). We also evaluated the LLM-based human interface in terms of time to execution and NASA TLX with human participant experiments. Results indicate that our ensemble-with-verification approach reliably maps operator intent to safe, executable multi-robot plans while maintaining low user effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17575v1">DeTAILS: Deep Thematic Analysis with Iterative LLM Support</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Thematic analysis is widely used in qualitative research but can be difficult to scale because of its iterative, interpretive demands. We introduce DeTAILS, a toolkit that integrates large language model (LLM) assistance into a workflow inspired by Braun and Clarke's thematic analysis framework. DeTAILS supports researchers in generating and refining codes, reviewing clusters, and synthesizing themes through interactive feedback loops designed to preserve analytic agency. We evaluated the system with 18 qualitative researchers analyzing Reddit data. Quantitative results showed strong alignment between LLM-supported outputs and participants' refinements, alongside reduced workload and high perceived usefulness. Qualitatively, participants reported that DeTAILS accelerated analysis, prompted reflexive engagement with AI outputs, and fostered trust through transparency and control. We contribute: (1) an interactive human-LLM workflow for large-scale qualitative analysis, (2) empirical evidence of its feasibility and researcher experience, and (3) design implications for trustworthy AI-assisted qualitative research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17535v1">How role-play shapes relevance judgment in zero-shot LLM rankers</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising zero-shot rankers, but their performance is highly sensitive to prompt formulation. In particular, role-play prompts, where the model is assigned a functional role or identity, often give more robust and accurate relevance rankings. However, the mechanisms and diversity of role-play effects remain underexplored, limiting both effective use and interpretability. In this work, we systematically examine how role-play variations influence zero-shot LLM rankers. We employ causal intervention techniques from mechanistic interpretability to trace how role-play information shapes relevance judgments in LLMs. Our analysis reveals that (1) careful formulation of role descriptions have a large effect on the ranking quality of the LLM; (2) role-play signals are predominantly encoded in early layers and communicate with task instructions in middle layers, while receiving limited interaction with query or document representations. Specifically, we identify a group of attention heads that encode information critical for role-conditioned relevance. These findings not only shed light on the inner workings of role-play in LLM ranking but also offer guidance for designing more effective prompts in IR and beyond, pointing toward broader opportunities for leveraging role-play in zero-shot applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12024v2">FlexQuant: A Flexible and Efficient Dynamic Precision Switching Framework for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 1p pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has exacerbated the memory bottleneck due to the widening gap between model parameter scaling and hardware capabilities. While post-training quantization techniques effectively reduce memory overhead, existing methods predominantly rely on static quantization strategies, which struggle to adapt to dynamic workloads. To address this, we propose FlexQuant, a dynamic precision-switching framework that optimizes the trade-off between inference speed and accuracy. Leveraging model perplexity entropy and Kullback-Leibler divergence, FlexQuant enables fine-grained, layer-wise mixed-precision quantization and dynamically adjusts bit-widths during each token generation. FlexQuant provides a comprehensive analysis of quantization strategies, introduces a precision requirement model for optimal switching, and implements efficient fine-grained precision management. Evaluations demonstrate that FlexQuant achieves a 1.3x end-to-end speedup across diverse language tasks with negligible accuracy loss introduced. This framework offers a flexible and adaptive solution for efficient LLM deployment. Code is released at https://github.com/ZongwuWang/FlexQuant.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17534v1">NieNie: Adaptive Rhythmic System for Stress Relief with LLM-Based Guidance</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Today's young people are facing increasing psychological stress due to various social issues. Traditional stress management tools often rely on static scripts or passive content, which are ineffective in alleviating stress. NieNie addresses this gap by combining rhythm biofeedback with real-time psychological guidance through a large language model (LLM), offering an interactive, tactile response. The system is specifically designed for young people experiencing emotional stress, collecting physiological signals such as heart rate variability and generating adaptive squeeze-release rhythms via soft, tactile devices. Utilising LLM, the system provides timely squeezing rhythms and psychologically guided feedback prompts, offering personalised rhythm games while reinforcing stress restructuring. Unlike traditional mental health apps, NieNie places users within an embodied interactive loop, leveraging tactile interaction, biofeedback, and adaptive language support to create an immersive stress regulation experience. This study demonstrates how embodied systems can connect bodily actions with mental health in everyday contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17532v1">OncoReason: Structuring Clinical Reasoning in LLMs for Robust and Interpretable Survival Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Predicting cancer treatment outcomes requires models that are both accurate and interpretable, particularly in the presence of heterogeneous clinical data. While large language models (LLMs) have shown strong performance in biomedical NLP, they often lack structured reasoning capabilities critical for high-stakes decision support. We present a unified, multi-task learning framework that aligns autoregressive LLMs with clinical reasoning for outcome prediction on the MSK-CHORD dataset. Our models are trained to jointly perform binary survival classification, continuous survival time regression, and natural language rationale generation. We evaluate three alignment strategies: (1) standard supervised fine-tuning (SFT), (2) SFT with Chain-of-Thought (CoT) prompting to elicit step-by-step reasoning, and (3) Group Relative Policy Optimization (GRPO), a reinforcement learning method that aligns model outputs to expert-derived reasoning trajectories. Experiments with LLaMa3-8B and Med42-8B backbones demonstrate that CoT prompting improves F1 by +6.0 and reduces MAE by 12%, while GRPO achieves state-of-the-art interpretability and predictive performance across BLEU, ROUGE, and BERTScore. We further show that existing biomedical LLMs often fail to produce valid reasoning traces due to architectural constraints. Our findings underscore the importance of reasoning-aware alignment in multi-task clinical modeling and set a new benchmark for interpretable, trustworthy LLMs in precision oncology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00161v2">Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution. In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision. For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation. Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17491v1">Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17476v1">Disparities in Multilingual LLM-Based Healthcare Q&A</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Equitable access to reliable health information is vital when integrating AI into healthcare. Yet, information quality varies across languages, raising concerns about the reliability and consistency of multilingual Large Language Models (LLMs). We systematically examine cross-lingual disparities in pre-training source and factuality alignment in LLM answers for multilingual healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and Italian. We (i) constructed Multilingual Wiki Health Care (MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed cross-lingual healthcare coverage; (iii) assessed LLM response alignment with these references; and (iv) conducted a case study on factual alignment through the use of contextual information and Retrieval-Augmented Generation (RAG). Our findings reveal substantial cross-lingual disparities in both Wikipedia coverage and LLM factual alignment. Across LLMs, responses align more with English Wikipedia, even when the prompts are non-English. Providing contextual excerpts from non-English Wikipedia at inference time effectively shifts factual alignment toward culturally relevant knowledge. These results highlight practical pathways for building more equitable, multilingual AI systems for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17472v1">Certified Self-Consistency: Statistical Guarantees and Test-Time Training for Reliable Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Recent advances such as self-consistency and test-time reinforcement learning (TTRL) improve the reliability of large language models (LLMs) without additional supervision, yet their underlying mechanisms and statistical guarantees remain poorly understood. We present a unified framework for certifiable inference in LLMs, showing that majority voting provides a statistical certificate of self-consistency: under mild assumptions, the aggregated answer coincides with the mode of the model's terminal distribution with high probability. We derive finite-sample and anytime-valid concentration bounds that quantify this confidence, and introduce the Martingale Majority Certificate (MMC), a sequential stopping rule that adaptively determines when sufficient samples have been drawn. We further prove that label-free post-training methods such as TTRL implicitly sharpen the answer distribution by exponentially tilting it toward its mode, thereby reducing the number of samples required for certification. Building on this insight, we propose new post-training objectives that explicitly optimise this trade-off between sharpness and bias. Together, these results explain and connect two central test-time scaling strategies, self-consistency and TTRL, within a single statistical framework for label-free, certifiable reliability in reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15188v2">OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 This is the authors' extended version of the paper accepted for publication at the ACM SIGSAC Conference on Computer and Communications Security (CCS 2025). The final published version is available at https://doi.org/10.1145/3719027.3765219
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03313v3">LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v4">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17389v1">EduAdapt: A Question Answer Benchmark Dataset for Evaluating Grade-Level Adaptability in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 28 pages, 2 figures, 14 tables, 50 listings, EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming education by answering questions, explaining complex concepts, and generating content across a wide range of subjects. Despite strong performance on academic benchmarks, they often fail to tailor responses to students' grade levels. This is a critical need in K-12 education, where age-appropriate vocabulary and explanation are essential for effective learning. Existing models frequently produce outputs that are too advanced or vague for younger learners, and there are no standardized benchmarks to evaluate their ability to adjust across cognitive and developmental stages. To address this gap, we introduce EduAdapt, a benchmark of nearly 48k grade-labeled QA pairs across nine science subjects, spanning Grades 1-12 and grouped into four grade levels. We evaluate a diverse set of open-source LLMs on EduAdapt and find that while larger models generally perform better, they still struggle with generating suitable responses for early-grade students (Grades 1-5). Our work presents the first dataset and evaluation framework for assessing grade-level adaptability in LLMs, aiming to foster more developmentally aligned educational AI systems through better training and prompting strategies. EduAdapt code and datasets are publicly available at https://github.com/NaumanNaeem/EduAdapt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17388v1">The Atomic Instruction Gap: Instruction-Tuned LLMs Struggle with Simple, Self-Contained Directives</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 11 pages, 1 figure, 8 tables
    </div>
    <details class="paper-abstract">
      Instruction-tuned large language models (IT-LLMs) exhibit strong zero-shot reasoning, yet their ability to execute simple, self-contained instructions remains underexplored, despite this being foundational to complex instruction-following. We evaluate 20 IT-LLMs on modified MMLU and MMLU-Pro benchmarks, by systematically varying the format of option labels (alphabetic, numeric, Roman) while keeping their meaning identical under four paradigms, namely: (1) With explicit instructions, label changes cause large performance shifts (e.g., -30.45\% for Roman vs. numeric), revealing instruction-format bias. (2) Without instructions, performance drops further (up to -10.84\%) and label sensitivity intensifies, underscoring the role of explicit guidance. (3) When option contents are removed, models fail random-choice baselines except with numeric labels, suggesting weak adherence to atomic directives. (4) Three-shot exemplars yield no significant gains in robustness or fidelity, and generation analyses show persistent label errors, especially for non-numeric formats. Across model sizes, larger LLMs achieve higher accuracy but remain inconsistent in instruction adherence. These results expose the insufficiencies of current instruction-tuning paradigms and highlight the need for evaluation methods and training strategies that explicitly target atomic instruction-following.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17385v1">TabR1: Taming GRPO for tabular reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Tabular prediction has traditionally relied on gradient-boosted decision trees and specialized deep learning models, which excel within tasks but provide limited interpretability and weak transfer across tables. Reasoning large language models (LLMs) promise cross-task adaptability with trans- parent reasoning traces, yet their potential has not been fully realized for tabular data. This paper presents TabR1, the first reasoning LLM for tabular prediction with multi-step reasoning. At its core is Permutation Relative Policy Optimization (PRPO), a simple yet efficient reinforcement learning method that encodes column-permutation invariance as a structural prior. By construct- ing multiple label-preserving permutations per sample and estimating advantages both within and across permutations, PRPO transforms sparse rewards into dense learning signals and improves generalization. With limited supervision, PRPO activates the reasoning ability of LLMs for tabular prediction, enhancing few-shot and zero-shot performance as well as interpretability. Comprehensive experiments demonstrate that TabR1 achieves performance comparable to strong baselines under full-supervision fine-tuning. In the zero-shot setting, TabR1 approaches the performance of strong baselines under the 32-shot setting. Moreover, TabR1 (8B) substantially outperforms much larger LLMs across various tasks, achieving up to 53.17% improvement over DeepSeek-R1 (685B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01234v2">First Field-Trial Demonstration of L4 Autonomous Optical Network for Distributed AI Training Communication: An LLM-Powered Multi-AI-Agent Solution</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted by 51st European Conference on Optical Communication (ECOC 2025), paper W.02.01.177
    </div>
    <details class="paper-abstract">
      We demonstrate the first cross-domain cross-layer level-4 autonomous optical network via a multi-AI-agent system. Field trials show ~98% task completion rate across the distributed AI training lifecycle-3.2x higher than single agents using state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17364v1">Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) excel at understanding videos in-context, provided they have full access to the video when answering queries. However, these models face challenges in streaming scenarios where hour-long videos must be processed online, and questions need timely responses. In this work, we propose a training-free approach compatible with standard Video-LLMs, leveraging three key concepts: 1) LLM-informed selection of visual tokens to identify those that the LLM has attended to and contributed to its understanding of each short clip. Our attention-based selection allows us to discard up to ~95% of unimportant visual tokens with minimal performance loss; 2) Recurrent processing of past selected tokens to generate temporally coherent understanding of each processed clip; 3) Caption-based question answering for lightweight and accurate responses. Our method achieves state-of-the-art performance on streaming video benchmarks, striking a balance between efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17358v1">Localist LLMs with Recruitment Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      We present a novel framework for training large language models with continuously adjustable internal representations that span the full spectrum from localist (interpretable, rule-based) to distributed (generalizable, efficient) encodings. The key innovations are (1) a locality dial, a tunable parameter that dynamically controls the degree of localization during both training and inference without requiring model retraining, (2) an information-theoretic recruitment mechanism that adaptively allocates semantic blocks as needed, eliminating the requirement for complete domain knowledge at initialization, and (3) a hierarchical recruitment framework that extends capacity allocation to entire specialized LLMs, enabling multi-granularity architectural adaptation. This is achieved through group sparsity penalties on attention mechanisms, information-theoretic anchor design, dynamic rule injection, and principled recruitment criteria based on penalized likelihood with explicit units. We provide rigorous mathematical results establishing explicit threshold conditions under which attention provably concentrates on semantically relevant blocks at stationary points, with exact bounds on attention entropy and pointer fidelity. The hierarchical recruitment mechanism provides convergence guarantees at both the block level (fine-grained, within-LLM) and the LLM level (coarse-grained, cross-domain), ensuring the system discovers semantic partitions that balance model complexity against data encoding efficiency. This framework enables practitioners to continuously interpolate between interpretable and high-performance modes while adapting architectural capacity at multiple granularities, supporting applications in regulated domains requiring both transparency and capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16293v4">Robust LLM Training Infrastructure at ByteDance</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform and achieves 97% ETTR for a three-month training job on 9,600 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09211v2">DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 This paper was accepted to the EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17281v1">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17210v1">Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v4">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 45 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have attracted widespread attention for their remarkable capabilities in multimodal data understanding. Meanwhile, the rapid expansion of information services has led to a growing demand for AI-enabled wireless networks. The open-source DeepSeek models are famous for their innovative designs, such as large-scale pure RL and cost-efficient training, which make them well-suited for practical deployment in wireless networks. By integrating DeepSeek-style LLMs with wireless infrastructures, a synergistic opportunity arises: the DeepSeek-style LLMs enhance network optimization with strong reasoning and decision-making abilities, while wireless infrastructure enables the broad deployment of these models. Motivated by this convergence, this survey presents a comprehensive DeepSeek-inspired exploration of RL-based LLMs in the context of wireless networks. We begin by reviewing key techniques behind network optimization to establish a foundation for understanding DeepSeek-style LLM integration. Next, we examine recent advancements in RL-based LLMs, using DeepSeek models as a representative example. Building on this, we explore the synergy between the two domains, highlighting motivations, challenges, and potential solutions. Finally, we highlight emerging directions for integrating LLMs with wireless networks, such as quantum, on-device, and neural-symbolic LLM models, as well as embodied AI agents. Overall, this survey offers a comprehensive examination of the interplay between DeepSeek-style LLMs and wireless networks, demonstrating how these domains can mutually enhance each other to drive innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17173v1">Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted to the NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      We study a web-deployed, tool-augmented LLM health coach with real users. In a pilot with seven users (280 rated turns), offline policy evaluation (OPE) over factorized decision heads (Tool/Style) shows that a uniform heavy-tool policy raises average value on logs but harms specific subgroups, most notably low-health-literacy/high-self-efficacy users. A lightweight simulator with hidden archetypes further shows that adding a small early information-gain bonus reliably shortens trait identification and improves goal success and pass@3. Together, these early findings indicate an evaluation-first path to personalization: freeze the generator, learn subgroup-aware decision heads on typed rewards (objective tool outcomes and satisfaction), and always report per-archetype metrics to surface subgroup harms that averages obscure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17015v1">Justitia: Fair and Efficient Scheduling for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      In the era of Large Language Models (LLMs), it has been popular to launch a series of LLM inferences -- we call an LLM application -- to better solve real-world problems. When serving those applications in shared GPU servers, the schedulers are expected to attain fast application completions with guaranteed worst-case performance. However, mainstream LLM schedulers fail to behave well for LLM applications -- due to head-of-line blocking or over-constrained resource allocation. In this paper, we propose to serve LLM applications in a fair and also efficient manner. To this end, we design Justitia, a novel scheduler with three key techniques. First, given that memory is prevalently a bottleneck for mainstream inference frameworks like vLLM, Justitia models the service cost of LLM applications in a memory-centric manner. Meanwhile, it uses a simple neural network model to conduct light-weight and also accurate demand prediction. Moreover, Justitia adopts a virtual-time based fair queuing algorithm to reduce the overall performance with guaranteed worst-case delay. We have implemented Justitia atop vLLM, and experimental results involving diverse LLM applications show that it can substantially enhance the scheduling efficiency with fairness preserved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17013v1">DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v4">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v3">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17002v1">EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Circuit schematics play a crucial role in analog integrated circuit design, serving as the primary medium for human understanding and verification of circuit functionality. While recent large language model (LLM)-based approaches have shown promise in circuit topology generation and device sizing, most rely solely on textual representations such as SPICE netlists, which lack visual interpretability for circuit designers. To address this limitation, we propose EEschematic, an AI agent for automatic analog schematic generation based on a Multimodal Large Language Model (MLLM). EEschematic integrates textual, visual, and symbolic modalities to translate SPICE netlists into schematic diagrams represented in a human-editable format. The framework uses six analog substructure examples for few-shot placement and a Visual Chain-of-Thought (VCoT) strategy to iteratively refine placement and wiring, enhancing schematic clarity and symmetry. Experimental results on representative analog circuits, including a CMOS inverter, a five-transistor operational transconductance amplifier (5T-OTA), and a telescopic cascode amplifier, demonstrate that EEschematic produces schematics with high visual quality and structural correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17001v1">Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17000v1">Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19504v2">DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code is available at https://github.com/unites-lab/doge
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD). In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs are accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving the performance of the teacher model, student models distilled from the defensively generated outputs demonstrate catastrophically reduced performance, demonstrating DOGe as a practical safeguard against KD-based model imitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16978v1">Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit ({\Delta}Score = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity ({\Delta}Score = 3.4, d_z = 1.86), ranked-choice voting ({\Delta}Score = 2.4, d_z = 1.20), and token penalties ({\Delta}Score = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20548v2">$Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q_sharp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16943v1">Peering Inside the Black Box: Uncovering LLM Errors in Optimization Modelling through Component-Level Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to convert natural language descriptions into mathematical optimization formulations. Current evaluations often treat formulations as a whole, relying on coarse metrics like solution accuracy or runtime, which obscure structural or numerical errors. In this study, we present a comprehensive, component-level evaluation framework for LLM-generated formulations. Beyond the conventional optimality gap, our framework introduces metrics such as precision and recall for decision variables and constraints, constraint and objective root mean squared error (RMSE), and efficiency indicators based on token usage and latency. We evaluate GPT-5, LLaMA 3.1 Instruct, and DeepSeek Math across optimization problems of varying complexity under six prompting strategies. Results show that GPT-5 consistently outperforms other models, with chain-of-thought, self-consistency, and modular prompting proving most effective. Analysis indicates that solver performance depends primarily on high constraint recall and low constraint RMSE, which together ensure structural correctness and solution reliability. Constraint precision and decision variable metrics play secondary roles, while concise outputs enhance computational efficiency. These findings highlight three principles for NLP-to-optimization modeling: (i) Complete constraint coverage prevents violations, (ii) minimizing constraint RMSE ensures solver-level accuracy, and (iii) concise outputs improve computational efficiency. The proposed framework establishes a foundation for fine-grained, diagnostic evaluation of LLMs in optimization modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16933v1">Tutoring LLM into a Better CUDA Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in Euro-Par 2025: Parallel Processing, Part II, and is available online at https://doi.org/10.1007/978-3-031-99857-7_18
    </div>
    <details class="paper-abstract">
      Recent leaps in large language models (LLMs) caused a revolution in programming tools (like GitHub Copilot) that can help with code generation, debugging, and even performance optimization. In this paper, we focus on the capabilities of the most recent reasoning models to generate optimized CUDA code for predefined, well-known tasks. Our objective is to determine which types of code optimizations and parallel patterns the LLMs can perform by themselves and whether they can be improved by tutoring (providing more detailed hints and guidelines in the prompt). The generated solutions were evaluated both automatically (for correctness and speedup) and manually (code reviews) to provide a more detailed perspective. We also tried an interactive approach where the LLM can fix its previous mistakes within a session. The results indicate that LLMs are quite skilled coders; however, they require tutoring to reach optimized solutions provided by parallel computing experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16932v1">Prompt-MII: Meta-Learning Instruction Induction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16916v1">SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer promising capabilities for tackling complex reasoning tasks, including optimization problems. However, existing methods either rely on prompt engineering, which leads to poor generalization across problem types, or require costly supervised training. We introduce SolverLLM, a training-free framework that leverages test-time scaling to solve diverse optimization problems. Rather than solving directly, SolverLLM generates mathematical formulations and translates them into solver-ready code, guided by a novel Monte Carlo Tree Search (MCTS) strategy. To enhance the search process, we modify classical MCTS with (1) dynamic expansion for adaptive formulation generation, (2) prompt backpropagation to guide exploration via outcome-driven feedback, and (3) uncertainty backpropagation to incorporate reward reliability into decision-making. Experiments on six standard benchmark datasets demonstrate that SolverLLM outperforms both prompt-based and learning-based baselines, achieving strong generalization without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04774v2">Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This abstract was accepted to and presented at the "Multi-Agent Cooperative Systems and Swarm Robotics in the Era of Generative AI" (MACRAI) workshop at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
    </div>
    <details class="paper-abstract">
      Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16882v1">Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10407v2">PrediQL: Automated Testing of GraphQL APIs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 8 pages, two columns
    </div>
    <details class="paper-abstract">
      GraphQL's flexible query model and nested data dependencies expose APIs to complex, context-dependent vulnerabilities that are difficult to uncover using conventional testing tools. Existing fuzzers either rely on random payload generation or rigid mutation heuristics, failing to adapt to the dynamic structures of GraphQL schemas and responses. We present PrediQL, the first retrieval-augmented, LLM-guided fuzzer for GraphQL APIs. PrediQL combines large language model reasoning with adaptive feedback loops to generate semantically valid and diverse queries. It models the choice of fuzzing strategy as a multi-armed bandit problem, balancing exploration of new query structures with exploitation of past successes. To enhance efficiency, PrediQL retrieves and reuses execution traces, schema fragments, and prior errors, enabling self-correction and progressive learning across test iterations. Beyond input generation, PrediQL integrates a context-aware vulnerability detector that uses LLM reasoning to analyze responses, interpreting data values, error messages, and status codes to identify issues such as injection flaws, access-control bypasses, and information disclosure. Our evaluation across open-source and benchmark GraphQL APIs shows that PrediQL achieves significantly higher coverage and vulnerability discovery rates compared to state-of-the-art baselines. These results demonstrate that combining retrieval-augmented reasoning with adaptive fuzzing can transform API security testing from reactive enumeration to intelligent exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16819v1">Cross-Genre Authorship Attribution via LLM-Based Retrieve-and-Rerank</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Authorship attribution (AA) is the task of identifying the most likely author of a query document from a predefined set of candidate authors. We introduce a two-stage retrieve-and-rerank framework that finetunes LLMs for cross-genre AA. Unlike the field of information retrieval (IR), where retrieve-and-rerank is a de facto strategy, cross-genre AA systems must avoid relying on topical cues and instead learn to identify author-specific linguistic patterns that are independent of the text's subject matter (genre/domain/topic). Consequently, for the reranker, we demonstrate that training strategies commonly used in IR are fundamentally misaligned with cross-genre AA, leading to suboptimal behavior. To address this, we introduce a targeted data curation strategy that enables the reranker to effectively learn author-discriminative signals. Using our LLM-based retrieve-and-rerank pipeline, we achieve substantial gains of 22.3 and 34.4 absolute Success@8 points over the previous state-of-the-art on HIATUS's challenging HRS1 and HRS2 cross-genre AA benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16809v1">When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14345v2">LLM-Enhanced Black-Litterman Portfolio Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Presented at the CIKM 2025 Workshop on Financial AI (https://advancesinfinancialai.com/)
    </div>
    <details class="paper-abstract">
      The Black-Litterman model addresses the sensitivity issues of tra- ditional mean-variance optimization by incorporating investor views, but systematically generating these views remains a key challenge. This study proposes and validates a systematic frame- work that translates return forecasts and predictive uncertainty from Large Language Models (LLMs) into the core inputs for the Black-Litterman model: investor views and their confidence lev- els. Through a backtest on S&P 500 constituents, we demonstrate that portfolios driven by top-performing LLMs significantly out- perform traditional baselines in both absolute and risk-adjusted terms. Crucially, our analysis reveals that each LLM exhibits a dis- tinct and consistent investment style which is the primary driver of performance. We found that the selection of an LLM is therefore not a search for a single best forecaster, but a strategic choice of an investment style whose success is contingent on its alignment with the prevailing market regime. The source code and data are available at https://github.com/youngandbin/LLM-BLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16794v1">Black-box Optimization of LLM Outputs by Asking for Directions</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      We present a novel approach for attacking black-box large language models (LLMs) by exploiting their ability to express confidence in natural language. Existing black-box attacks require either access to continuous model outputs like logits or confidence scores (which are rarely available in practice), or rely on proxy signals from other models. Instead, we demonstrate how to prompt LLMs to express their internal confidence in a way that is sufficiently calibrated to enable effective adversarial optimization. We apply our general method to three attack scenarios: adversarial examples for vision-LLMs, jailbreaks and prompt injections. Our attacks successfully generate malicious inputs against systems that only expose textual outputs, thereby dramatically expanding the attack surface for deployed LLMs. We further find that better and larger models exhibit superior calibration when expressing confidence, creating a concerning security paradox where model capability improvements directly enhance vulnerability. Our code is available at this [link](https://github.com/zj-jayzhang/black_box_llm_optimization).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v6">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Project Page: https://awesome-llm-as-a-judge.github.io/
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01005v2">When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10213v2">An Empirical Study on LLM-based Agents for Automated Bug Fixing</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and LLM-based Agents have been applied to fix bugs automatically, demonstrating the capability in addressing software defects by engaging in development environment interaction, iterative validation and code modification. However, systematic analysis of these agent systems remain limited, particularly regarding performance variations among top-performing ones. In this paper, we examine six repair systems on the SWE-bench Verified benchmark for automated bug fixing. We first assess each system's overall performance, noting the instances solvable by all or none of these systems, and explore the capabilities of different systems. We also compare fault localization accuracy at file and code symbol levels and evaluate bug reproduction capabilities. Through analysis, we concluded that further optimization is needed in both the LLM capability itself and the design of Agentic flow to improve the effectiveness of the Agent in bug fixing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18439v2">A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted for publication in the 24th IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom 2025)
    </div>
    <details class="paper-abstract">
      Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable. This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient. A replication package is available for download from https://doi.org/10.5281/zenodo.17341503. Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16767v1">T3 Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Translating natural language instructions into executable motion plans is a fundamental challenge in robotics. Traditional approaches are typically constrained by their reliance on domain-specific expertise to customize planners, and often struggle with spatio-temporal couplings that usually lead to infeasible motions or discrepancies between task planning and motion execution. Despite the proficiency of Large Language Models (LLMs) in high-level semantic reasoning, hallucination could result in infeasible motion plans. In this paper, we introduce the T3 Planner, an LLM-enabled robotic motion planning framework that self-corrects it output with formal methods. The framework decomposes spatio-temporal task constraints via three cascaded modules, each of which stimulates an LLM to generate candidate trajectory sequences and examines their feasibility via a Signal Temporal Logic (STL) verifier until one that satisfies complex spatial, temporal, and logical constraints is found.Experiments across different scenarios show that T3 Planner significantly outperforms the baselines. The required reasoning can be distilled into a lightweight Qwen3-4B model that enables efficient deployment. All supplementary materials are accessible at https://github.com/leeejia/T3_Planner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11924v2">Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction refers to the phenomenon where a language model refines its own outputs purely through prompting, without external feedback or parameter updates. While this approach improves performance across diverse tasks, its internal mechanism remains poorly understood. We analyze intrinsic self-correction from a representation-level perspective. We formalize and introduce the notion of a prompt-induced shift, which is the change in hidden representations caused by a self-correction prompt. Across 5 open-source LLMs, prompt-induced shifts in text detoxification and text toxification align with latent directions constructed from contrastive pairs. In detoxification, the shifts align with the non-toxic direction; in toxification, they align with the toxic direction. These results suggest that intrinsic self-correction functions as representation steering along interpretable latent directions, beyond what standard metrics such as task scores or model confidence capture. Our analysis offers an interpretability-based account of intrinsic self-correction and contributes to a more systematic understanding of LLM prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00641v2">AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This paper is accepted by 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Despite the rapid advancement of LLM-based agents, the reliable evaluation of their safety and security remains a significant challenge. Existing rule-based or LLM-based evaluators often miss dangers in agents' step-by-step actions, overlook subtle meanings, fail to see how small issues compound, and get confused by unclear safety or security rules. To overcome this evaluation crisis, we introduce AgentAuditor, a universal, training-free, memory-augmented reasoning framework that empowers LLM evaluators to emulate human expert evaluators. AgentAuditor constructs an experiential memory by having an LLM adaptively extract structured semantic features (e.g., scenario, risk, behavior) and generate associated chain-of-thought reasoning traces for past interactions. A multi-stage, context-aware retrieval-augmented generation process then dynamically retrieves the most relevant reasoning experiences to guide the LLM evaluator's assessment of new cases. Moreover, we developed ASSEBench, the first benchmark designed to check how well LLM-based evaluators can spot both safety risks and security threats. ASSEBench comprises 2293 meticulously annotated interaction records, covering 15 risk types across 29 application scenarios. A key feature of ASSEBench is its nuanced approach to ambiguous risk situations, employing "Strict" and "Lenient" judgment standards. Experiments demonstrate that AgentAuditor not only consistently improves the evaluation performance of LLMs across all benchmarks but also sets a new state-of-the-art in LLM-as-a-judge for agent safety and security, achieving human-level accuracy. Our work is openly accessible at https://github.com/Astarojth/AgentAuditor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16716v1">DistilLock: Safeguarding LLMs from Unauthorized Knowledge Distillation on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance across diverse tasks, but fine-tuning them typically relies on cloud-based, centralized infrastructures. This requires data owners to upload potentially sensitive data to external servers, raising serious privacy concerns. An alternative approach is to fine-tune LLMs directly on edge devices using local data; however, this introduces a new challenge: the model owner must transfer proprietary models to the edge, which risks intellectual property (IP) leakage. To address this dilemma, we propose DistilLock, a TEE-assisted fine-tuning framework that enables privacy-preserving knowledge distillation on the edge. In DistilLock, a proprietary foundation model is executed within a trusted execution environment (TEE) enclave on the data owner's device, acting as a secure black-box teacher. This setup preserves both data privacy and model IP by preventing direct access to model internals. Furthermore, DistilLock employs a model obfuscation mechanism to offload obfuscated weights to untrusted accelerators for efficient knowledge distillation without compromising security. We demonstrate that DistilLock prevents unauthorized knowledge distillation processes and model-stealing attacks while maintaining high computational efficiency, but offering a secure and practical solution for edge-based LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16713v1">so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Whitespace is a critical component of poetic form, reflecting both adherence to standardized forms and rebellion against those forms. Each poem's whitespace distribution reflects the artistic choices of the poet and is an integral semantic and spatial feature of the poem. Yet, despite the popularity of poetry as both a long-standing art form and as a generation task for large language models (LLMs), whitespace has not received sufficient attention from the NLP community. Using a corpus of 19k English-language published poems from Poetry Foundation, we investigate how 4k poets have used whitespace in their works. We release a subset of 2.8k public-domain poems with preserved formatting to facilitate further research in this area. We compare whitespace usage in the published poems to (1) 51k LLM-generated poems, and (2) 12k unpublished poems posted in an online community. We also explore whitespace usage across time periods, poetic forms, and data sources. Additionally, we find that different text processing methods can result in significantly different representations of whitespace in poetry data, motivating us to use these poems and whitespace patterns to discuss implications for the processing strategies used to assemble pretraining datasets for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16712v1">The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11108v2">A Vision for Access Control in LLM-based Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 11 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16701v1">An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18573v2">Enhancing Efficiency and Exploration in Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accept by EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      Reasoning large language models (LLMs) excel in complex tasks, which has drawn significant attention to reinforcement learning (RL) for LLMs. However, existing approaches allocate an equal number of rollouts to all questions during the RL process, which is inefficient. This inefficiency stems from the fact that training on simple questions yields limited gains, whereas more rollouts are needed for challenging questions to sample correct answers. Furthermore, while RL improves response precision, it limits the model's exploration ability, potentially resulting in a performance cap below that of the base model prior to RL. To address these issues, we propose a mechanism for dynamically allocating rollout budgets based on the difficulty of the problems, enabling more efficient RL training. Additionally, we introduce an adaptive dynamic temperature adjustment strategy to maintain the entropy at a stable level, thereby encouraging sufficient exploration. This enables LLMs to improve response precision while preserving their exploratory ability to uncover potential correct pathways. The code and data is available on: https://github.com/LiaoMengqi/E3-RL4LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16686v1">Investigating the Impact of Rationales for LLMs on Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) rationales, which provide step-by-step reasoning to derive final answers, benefit LLMs in both inference and training. Incorporating rationales, either by generating them before answering during inference, or by placing them before or after the original answers during training - significantly improves model performance on mathematical, symbolic and commonsense reasoning tasks. However, most work focuses on the role of rationales in these reasoning tasks, overlooking their potential impact on other important tasks like natural language understanding (NLU) tasks. In this work, we raise the question: Can rationales similarly benefit NLU tasks? To conduct a systematic exploration, we construct NLURC, a comprehensive and high-quality NLU dataset collection with rationales, and develop various rationale-augmented methods. Through exploring the applicability of these methods on NLU tasks using the dataset, we uncover several potentially surprising findings: (1) CoT inference shifts from hindering NLU performance to surpassing direct label prediction as model size grows, indicating a positive correlation. (2) Most rationale-augmented training methods perform worse than label-only training, with one specially designed method consistently achieving improvements. (3) LLMs trained with rationales achieve significant performance gains on unseen NLU tasks, rivaling models ten times their size, while delivering interpretability on par with commercial LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v3">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Published as a conference paper at Neurips 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17913v1">TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted for publication in the proceedings of ICTAI 2025
    </div>
    <details class="paper-abstract">
      Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17910v1">Interpretability Framework for LLMs in Undergraduate Calculus</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used in education, yet their correctness alone does not capture the quality, reliability, or pedagogical validity of their problem-solving behavior, especially in mathematics, where multistep logic, symbolic reasoning, and conceptual clarity are critical. Conventional evaluation methods largely focus on final answer accuracy and overlook the reasoning process. To address this gap, we introduce a novel interpretability framework for analyzing LLM-generated solutions using undergraduate calculus problems as a representative domain. Our approach combines reasoning flow extraction and decomposing solutions into semantically labeled operations and concepts with prompt ablation analysis to assess input salience and output stability. Using structured metrics such as reasoning complexity, phrase sensitivity, and robustness, we evaluated the model behavior on real Calculus I to III university exams. Our findings revealed that LLMs often produce syntactically fluent yet conceptually flawed solutions, with reasoning patterns sensitive to prompt phrasing and input variation. This framework enables fine-grained diagnosis of reasoning failures, supports curriculum alignment, and informs the design of interpretable AI-assisted feedback tools. This is the first study to offer a structured, quantitative, and pedagogically grounded framework for interpreting LLM reasoning in mathematics education, laying the foundation for the transparent and responsible deployment of AI in STEM learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17904v1">BreakFun: Jailbreaking LLMs via Schema Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17902v1">Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Model (LLM) architectures presents a fundamental challenge: valuable, task-specific behaviors learned through fine-tuning methods like Low-Rank Adaptation (LoRA) are effectively trapped within their source model's architecture, herein referred to architectural lock-in. Existing transfer methods attempt to bridge this gap by aligning the static weight spaces of models, a brittle and indirect approach that relies on tenuous correlations between parameter geometries. This paper introduces a fundamentally different and more direct paradigm: the Cartridge Activation Space Transfer (CAST), a novel framework that liberates LoRA-encoded behaviors by learning a direct, nonlinear mapping between the activation manifolds, the geometric structures formed by the model's internal neuron activations, of two distinct LLM architectures. CAST treats a pre-trained LoRA as a frozen "behavioral kernel." It learns a set of lightweight, bidirectional projection heads that translate the target model's activation stream into the source model's latent space, apply the frozen kernel, and project the result back. This process, trained on a general text corpus without any task-specific data, effectively decouples the learned skill from the source architecture. We demonstrate that CAST enables true "zero-shot" translation of any standard LoRA adapter. Our experiments, including transfers between heterogeneous model families like Llama-2 and Mistral, show that CAST-translated adapters achieve 85-95\% of the performance of a LoRA fully retrained on the target model, quantitatively outperforming current weight-space transfer techniques and establishing a new state-of-the-art in model interoperability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17900v1">Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are entering legal workflows, yet we lack a jurisdiction-specific framework to assess their baseline competence therein. We use India's public legal examinations as a transparent proxy. Our multi-year benchmark assembles objective screens from top national and state exams and evaluates open and frontier LLMs under real-world exam conditions. To probe beyond multiple-choice questions, we also include a lawyer-graded, paired-blinded study of long-form answers from the Supreme Court's Advocate-on-Record exam. This is, to our knowledge, the first exam-grounded, India-specific yardstick for LLM court-readiness released with datasets and protocols. Our work shows that while frontier systems consistently clear historical cutoffs and often match or exceed recent top-scorer bands on objective exams, none surpasses the human topper on long-form reasoning. Grader notes converge on three reliability failure modes: procedural or format compliance, authority or citation discipline, and forum-appropriate voice and structure. These findings delineate where LLMs can assist (checks, cross-statute consistency, statute and precedent lookups) and where human leadership remains essential: forum-specific drafting and filing, procedural and relief strategy, reconciling authorities and exceptions, and ethical, accountable judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17023v1">Enrich and Detect: Video Temporal Grounding with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 ICCV 2025 (Highlights)
    </div>
    <details class="paper-abstract">
      We introduce ED-VTG, a method for fine-grained video temporal grounding utilizing multi-modal large language models. Our approach harnesses the capabilities of multimodal LLMs to jointly process text and video, in order to effectively localize natural language queries in videos through a two-stage process. Rather than being directly grounded, language queries are initially transformed into enriched sentences that incorporate missing details and cues to aid in grounding. In the second stage, these enriched queries are grounded, using a lightweight decoder, which specializes at predicting accurate boundaries conditioned on contextualized representations of the enriched queries. To mitigate noise and reduce the impact of hallucinations, our model is trained with a multiple-instance-learning objective that dynamically selects the optimal version of the query for each training sample. We demonstrate state-of-the-art results across various benchmarks in temporal video grounding and paragraph grounding settings. Experiments reveal that our method significantly outperforms all previously proposed LLM-based temporal grounding approaches and is either superior or comparable to specialized models, while maintaining a clear advantage against them in zero-shot evaluation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17021v1">Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v1">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code: https://github.com/ZQS1943/SafeSearch
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16590v1">Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Alan Kai Hassen and Andrius Bernatavicius contributed equally to this work
    </div>
    <details class="paper-abstract">
      Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring labeled training data. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a one-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation. We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq90\%$), named reaction classes ($\geq40\%$), and final reactants ($\geq74\%$). Beyond solving complex chemical tasks, our work also provides a method to generate theoretically grounded synthetic datasets by mapping chemical knowledge onto the molecular structure and thereby addressing data scarcity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v4">Unlocking LLM Repair Capabilities Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06767v2">Beyond Surface Similarity: Evaluating LLM-Based Test Refactorings with Structural and Semantic Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to refactor unit tests, improving readability and structure while preserving behavior. Evaluating such refactorings, however, remains difficult: metrics like CodeBLEU penalize beneficial renamings and edits, while semantic similarities overlook readability and modularity. We propose CTSES, a first step toward human-aligned evaluation of refactored tests. CTSES combines CodeBLEU, METEOR, and ROUGE-L into a composite score that balances semantics, lexical clarity, and structural alignment. Evaluated on 5,000+ refactorings from Defects4J and SF110 (GPT-4o and Mistral-Large), CTSES reduces false negatives and provides more interpretable signals than individual metrics. Our emerging results illustrate that CTSES offers a proof-of-concept for composite approaches, showing their promise in bridging automated metrics and developer judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22961v2">ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16559v1">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
