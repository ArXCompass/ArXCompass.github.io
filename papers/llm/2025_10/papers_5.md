# llm - 2025_10

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08670v2">Augmenting Smart Contract Decompiler Output through Fine-grained Dependency Analysis and LLM-facilitated Semantic Recovery</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 This is the author version of the article accepted for publication in IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Decompiler is a specialized type of reverse engineering tool extensively employed in program analysis tasks, particularly in program comprehension and vulnerability detection. However, current Solidity smart contract decompilers face significant limitations in reconstructing the original source code. In particular, the bottleneck of SOTA decompilers lies in inaccurate method identification, incorrect variable type recovery, and missing contract attributes. These deficiencies hinder downstream tasks and understanding of the program logic. To address these challenges, we propose SmartHalo, a new framework that enhances decompiler output by combining static analysis (SA) and large language models (LLM). SmartHalo leverages the complementary strengths of SA's accuracy in control and data flow analysis and LLM's capability in semantic prediction. More specifically, \system{} constructs a new data structure - Dependency Graph (DG), to extract semantic dependencies via static analysis. Then, it takes DG to create prompts for LLM optimization. Finally, the correctness of LLM outputs is validated through symbolic execution and formal verification. Evaluation on a dataset consisting of 465 randomly selected smart contract methods shows that SmartHalo significantly improves the quality of the decompiled code, compared to SOTA decompilers (e.g., Gigahorse). Notably, integrating GPT-4o with SmartHalo further enhances its performance, achieving precision rates of 87.39% for method boundaries, 90.39% for variable types, and 80.65% for contract attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v3">The simulation of judgment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Please refer to published version: https://doi.org/10.1073/pnas.2518443122
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in evaluative processes, from information filtering to assessing and addressing knowledge gaps through explanation and credibility judgments. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check--and against human judgments collected through a controlled experiment. We use news domains purely as a controlled benchmark for evaluative tasks, focusing on the underlying mechanisms rather than on news classification per se. To enable direct comparison, we implement a structured agentic framework in which both models and nonexpert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, our findings show consistent differences in the observable criteria guiding model evaluations, suggesting that lexical associations and statistical priors could influence evaluations in ways that differ from contextual reasoning. This reliance is associated with systematic effects: political asymmetries and a tendency to confuse linguistic form with epistemic reliability--a dynamic we term epistemia, the illusion of knowledge that emerges when surface plausibility replaces verification. Indeed, delegating judgment to such systems may affect the heuristics underlying evaluative processes, suggesting a shift from normative reasoning toward pattern-based approximation and raising open questions about the role of LLMs in evaluative processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14660v1">An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12350v2">O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14629v1">MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The application of Large Language Models (LLMs) in recommender systems faces key challenges in delivering deep personalization and intelligent reasoning, especially for interactive scenarios. Current methods are often constrained by limited context windows and single-turn reasoning, hindering their ability to capture dynamic user preferences and proactively reason over recommendation contexts. To address these limitations, we propose MR.Rec, a novel framework that synergizes memory and reasoning for LLM-based recommendations. To achieve personalization, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Furthermore, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoning enhanced memory retrieval. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective strategies for both memory utilization and reasoning refinement. By combining dynamic memory retrieval with adaptive reasoning, this approach ensures more accurate, context-aware, and highly personalized recommendations. Extensive experiments demonstrate that MR.Rec significantly outperforms state-of-the-art baselines across multiple metrics, validating its efficacy in delivering intelligent and personalized recommendations. We will release code and data upon paper notification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14628v1">RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22544v2">HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The submission was made prematurely. The authors plan to resubmit under the supervision of the corresponding author
    </div>
    <details class="paper-abstract">
      Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v3">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13193v2">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at https://github.com/kilgrims/ReMindRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14565v1">Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14548v1">LLM Agents Beyond Utility: An Open-Ended Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04575v2">Are LLMs Stable Formal Logic Translators in Logical Reasoning Across Linguistically Diversified Texts?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Logical reasoning with large language models (LLMs) has received growing attention. One mainstream approach translates natural language into formal logic and then applies symbolic solvers for deduction. While effective in many tasks, these LLM-based translators often fail to generate consistent symbolic representations when the same concept appears in different linguistic forms. Such inconsistencies break logical coherence and lead to solver errors. However, most existing benchmarks lack this type of linguistic variation, which frequently occurs in real-world text, leaving the problem underexplored. To address this gap, we present SoLT, a benchmark that systematically rewrites reasoning datasets into diverse yet logically equivalent forms across multiple levels. Beyond evaluation, SoLT also provides a general method to enrich any dataset with linguistic diversity while preserving both meaning and logic. To further enhance the stability of LLM-based reasoning, we propose MenTaL, which explicitly guides models to build a concept-symbol mapping table during translation. By linking equivalent expressions to shared symbols, MenTaL maintains consistency and mitigates symbol drift. Experiments on SoLT demonstrate that LLMs indeed suffer from inconsistent symbol mapping under linguistic variation, leading to significant drops in reasoning accuracy. Meanwhile, applying MenTaL brings clear and stable performance improvements across diverse inputs. Overall, our findings reveal that overlooking linguistic diversity hides key weaknesses in LLM-based translators, and our work offers a step toward more reliable logical reasoning in varied real-world scenarios. Our code is available at https://github.com/wufeiwuwoshihua/LinguDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14522v1">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03867v3">Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted for oral presentation at the EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03479v4">Women, Infamous, and Exotic Beings: A Comparative Study of Honorific Usages in Wikipedia and LLMs for Bengali and Hindi</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted and published at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      The obligatory use of third-person honorifics is a distinctive feature of several South Asian languages, encoding nuanced socio-pragmatic cues such as power, age, gender, fame, and social distance. In this work, (i) We present the first large-scale study of third-person honorific pronoun and verb usage across 10,000 Hindi and Bengali Wikipedia articles with annotations linked to key socio-demographic attributes of the subjects, including gender, age group, fame, and cultural origin. (ii) Our analysis uncovers systematic intra-language regularities but notable cross-linguistic differences: honorifics are more prevalent in Bengali than in Hindi, while non-honorifics dominate while referring to infamous, juvenile, and culturally exotic entities. Notably, in both languages, and more prominently in Hindi, men are more frequently addressed with honorifics than women. (iii) To examine whether large language models (LLMs) internalize similar socio-pragmatic norms, we probe six LLMs using controlled generation and translation tasks over 1,000 culturally balanced entities. We find that LLMs diverge from Wikipedia usage, exhibiting alternative preferences in honorific selection across tasks, languages, and socio-demographic attributes. These discrepancies highlight gaps in the socio-cultural alignment of LLMs and open new directions for studying how LLMs acquire, adapt, or distort social-linguistic norms. Our code and data are publicly available at https://github.com/souro/honorific-wiki-llm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14444v1">A Free Lunch in LLM Compression: Revisiting Retraining after Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12217v2">HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v2">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v2">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14401v1">The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14392v1">FairBatching: Fairness-Aware Batch Formation for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference systems face a fundamental tension between minimizing Time-to-First-Token (TTFT) latency for new requests and maintaining a high, steady token generation rate (low Time-Per-Output-Token, or TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by Sarathi, while effective at preventing decode stalls, introduce significant computational unfairness. They prioritize decode tasks excessively, simultaneously leading to underutilized decode slack and unnecessary prefill queuing delays, which collectively degrade the system's overall quality of service (QoS). This work identifies the root cause of this unfairness: the non-monotonic nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid decode-prioritizing policy that fails to adapt to dynamic workload bursts. We therefore propose FairBatching, a novel LLM inference scheduler that enforces fair resource allocation between prefill and decode tasks. It features an adaptive batch capacity determination mechanism, which dynamically adjusts the computational budget to improve the GPU utilization without triggering SLO violations. Its fair and dynamic batch formation algorithm breaks away from the decode-prioritizing paradigm, allowing computation resources to be reclaimed from bursting decode tasks to serve prefill surges, achieving global fairness. Furthermore, FairBatching provides a novel load estimation method, enabling more effective coordination with upper-level schedulers. Implemented and evaluated on realistic traces, FairBatching significantly reduces TTFT tail latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall 20.0% improvement in single-node capacity and 54.3% improvement in cluster-level capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14387v1">Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14381v1">Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13903v2">CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025 Conference
    </div>
    <details class="paper-abstract">
      Proprietary large language models (LLMs) exhibit strong generalization capabilities across diverse tasks and are increasingly deployed on edge devices for efficiency and privacy reasons. However, deploying proprietary LLMs at the edge without adequate protection introduces critical security threats. Attackers can extract model weights and architectures, enabling unauthorized copying and misuse. Even when protective measures prevent full extraction of model weights, attackers may still perform advanced attacks, such as fine-tuning, to further exploit the model. Existing defenses against these threats typically incur significant computational and communication overhead, making them impractical for edge deployment. To safeguard the edge-deployed LLMs, we introduce CoreGuard, a computation- and communication-efficient protection method. CoreGuard employs an efficient protection protocol to reduce computational overhead and minimize communication overhead via a propagation protocol. Extensive experiments show that CoreGuard achieves upper-bound security protection with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14365v1">On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce \nameshort{}, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and \textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14845v3">AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at NLP4PI @ EMNLP 2024
    </div>
    <details class="paper-abstract">
      Detecting biases in natural language understanding (NLU) for African American Vernacular English (AAVE) is crucial to developing inclusive natural language processing (NLP) systems. To address dialect-induced performance discrepancies, we introduce AAVENUE ({AAVE} {N}atural Language {U}nderstanding {E}valuation), a benchmark for evaluating large language model (LLM) performance on NLU tasks in AAVE and Standard American English (SAE). AAVENUE builds upon and extends existing benchmarks like VALUE, replacing deterministic syntactic and morphological transformations with a more flexible methodology leveraging LLM-based translation with few-shot prompting, improving performance across our evaluation metrics when translating key tasks from the GLUE and SuperGLUE benchmarks. We compare AAVENUE and VALUE translations using five popular LLMs and a comprehensive set of metrics including fluency, BARTScore, quality, coherence, and understandability. Additionally, we recruit fluent AAVE speakers to validate our translations for authenticity. Our evaluations reveal that LLMs consistently perform better on SAE tasks than AAVE-translated versions, underscoring inherent biases and highlighting the need for more inclusive NLP models. We have open-sourced our source code on GitHub and created a website to showcase our work at https://aavenuee.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07452v2">When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14348v1">Automated Extraction of Protocol State Machines from 3GPP Specifications with Domain-Informed Prompts and LLM Ensembles</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mobile telecommunication networks are foundational to global infrastructure and increasingly support critical sectors such as manufacturing, transportation, and healthcare. The security and reliability of these networks are essential, yet depend heavily on accurate modeling of underlying protocols through state machines. While most prior work constructs such models manually from 3GPP specifications, this process is labor-intensive, error-prone, and difficult to maintain due to the complexity and frequent updates of the specifications. Recent efforts using natural language processing have shown promise, but remain limited in handling the scale and intricacy of cellular protocols. In this work, we propose SpecGPT, a novel framework that leverages large language models (LLMs) to automatically extract protocol state machines from 3GPP documents. SpecGPT segments technical specifications into meaningful paragraphs, applies domain-informed prompting with chain-of-thought reasoning, and employs ensemble methods to enhance output reliability. We evaluate SpecGPT on three representative 5G protocols (NAS, NGAP, and PFCP) using manually annotated ground truth, and show that it outperforms existing approaches, demonstrating the effectiveness of LLMs for protocol modeling at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04384v2">LLM Based Bayesian Optimization for Prompt Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14331v1">LLM-ERM: Sample-Efficient Program Learning via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We seek algorithms for program learning that are both sample-efficient and computationally feasible. Classical results show that targets admitting short program descriptions (e.g., with short ``python code'') can be learned with a ``small'' number of examples (scaling with the size of the code) via length-first program enumeration, but the search is exponential in description length. Consequently, Gradient-based training avoids this cost yet can require exponentially many samples on certain short-program families. To address this gap, we introduce LLM-ERM, a propose-and-verify framework that replaces exhaustive enumeration with an LLM-guided search over candidate programs while retaining ERM-style selection on held-out data. Specifically, we draw $k$ candidates with a pretrained reasoning-augmented LLM, compile and check each on the data, and return the best verified hypothesis, with no feedback, adaptivity, or gradients. Theoretically, we show that coordinate-wise online mini-batch SGD requires many samples to learn certain short programs. {\em Empirically, LLM-ERM solves tasks such as parity variants, pattern matching, and primality testing with as few as 200 samples, while SGD-trained transformers overfit even with 100,000 samples}. These results indicate that language-guided program synthesis recovers much of the statistical efficiency of finite-class ERM while remaining computationally tractable, offering a practical route to learning succinct hypotheses beyond the reach of gradient-based training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14471v2">Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted as a spotlight at the 1st workshop on Multilingual and Equitable Language Technologies (MELT), COLM 2025
    </div>
    <details class="paper-abstract">
      Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21638v3">R1-Ranker: Teaching LLM Rankers to Reason</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong reasoning abilities in domains like mathematics, coding, and scientific problem-solving, yet their potential for ranking tasks, where prime examples include retrieval, recommender systems, and LLM routing, remains underexplored. Ranking requires complex reasoning across heterogeneous candidates, but existing LLM-based rankers are often domain-specific, tied to fixed backbones, and lack iterative refinement, limiting their ability to fully exploit LLMs' reasoning potential. To address these challenges, we propose R1-Ranker, a reasoning-incentive framework built on reinforcement learning, with two complementary designs: DRanker, which generates full rankings in one shot, and IRanker, which decomposes ranking into an iterative elimination process with step-wise rewards to encourage deeper reasoning. We evaluate unified R1-Rankers on nine datasets spanning recommendation, routing, and passage ranking, showing that IRanker-3B consistently achieves state-of-the-art performance, surpasses larger 7B models on some tasks, and yields a 15.7% average relative improvement. Ablation and generalization experiments further confirm the critical role of reinforcement learning and iterative reasoning, with IRanker-3B improving zero-shot performance by over 9% on out-of-domain tasks and reasoning traces boosting other LLMs by up to 22.87%. These results demonstrate that unifying diverse ranking tasks with a single reasoning-driven foundation model is both effective and essential for advancing LLM reasoning in ranking scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14278v1">PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14277v1">GenLARP: Enabling Immersive Live Action Role-Play through LLM-Generated Worlds and Characters</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We introduce GenLARP, a virtual reality (VR) system that transforms personalized stories into immersive live action role-playing (LARP) experiences. GenLARP enables users to act as both creators and players, allowing them to design characters based on their descriptions and live in the story world. Generative AI and agents powered by Large Language Models (LLMs) enrich these experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08615v3">Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mathematical reasoning serves as a crucial testbed for the intelligence of large language models (LLMs), and math word problems (MWPs) are a popular type of math problems. Most MWP datasets consist of problems containing only the necessary information, while problems with distracting and excessive conditions are often overlooked. Prior works have tested popular LLMs and found a dramatic performance drop in the presence of distracting conditions. However, datasets of MWPs with distracting conditions are limited, and most suffer from lower levels of difficulty and out-of-context expressions. This makes distracting conditions easy to identify and exclude, thus reducing the credibility of benchmarking on them. Moreover, when adding distracting conditions, the reasoning and answers may also change, requiring intensive labor to check and write the solutions. To address these issues, we design an iterative framework to generate distracting conditions using LLMs. We develop a set of prompts to revise MWPs from different perspectives and cognitive levels, encouraging the generation of distracting conditions as well as suggestions for further revision. Another advantage is the shared solutions between original and revised problems: we explicitly guide the LLMs to generate distracting conditions that do not alter the original solutions, thus avoiding the need to generate new solutions. This framework is efficient and easy to deploy, reducing the overhead of generating MWPs with distracting conditions while maintaining data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14253v1">Towards Agentic Self-Learning LLMs in Search Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://github.com/forangel2014/Towards-Agentic-Self-Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13921v2">APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at https://github.com/hwj20/APEX_EXP .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v2">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14242v1">Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 14 pages, 6 figures, 3 tables, and 1 algorithm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce inconsistent answers when faced with different phrasings of the same prompt. In this paper, we propose Flip-Flop Consistency ($F^2C$), an unsupervised training method that improves robustness to such perturbations. $F^2C$ is composed of two key components. The first, Consensus Cross-Entropy (CCE), uses a majority vote across prompt variations to create a hard pseudo-label. The second is a representation alignment loss that pulls lower-confidence and non-majority predictors toward the consensus established by high-confidence, majority-voting variations. We evaluate our method on 11 datasets spanning four NLP tasks, with 4-15 prompt variations per dataset. On average, $F^2C$ raises observed agreement by 11.62%, improves mean $F_1$ by 8.94%, and reduces performance variance across formats by 3.29%. In out-of-domain evaluations, $F^2C$ generalizes effectively, increasing $\overline{F_1}$ and agreement while decreasing variance across most source-target pairs. Finally, when trained on only a subset of prompt perturbations and evaluated on held-out formats, $F^2C$ consistently improves both performance and agreement while reducing variance. These findings highlight $F^2C$ as an effective unsupervised method for enhancing LLM consistency, performance, and generalization under prompt perturbations. Code is available at https://github.com/ParsaHejabi/Flip-Flop-Consistency-Unsupervised-Training-for-Robustness-to-Prompt-Perturbations-in-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00527v2">Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by IEEE/RSJ IROS 2025
    </div>
    <details class="paper-abstract">
      The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v4">LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 11 pages, 6 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. In this paper, we investigate the Soft Thinking capabilities of various LLMs through a systematic analysis of their internal behavior using a suite of probing techniques. Contrary to the prevailing belief that Soft Thinking supports parallel exploration of diverse reasoning paths, our findings reveal that LLMs behave as single-threaded reasoners--they predominantly rely on the token with the highest probability in the soft input to predict the next step. This behavior induces a greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting richer information via Soft Tokens. To address this Greedy Pitfall, we propose Stochastic Soft Thinking, which introduces stochasticity to break free from this Greedy Pitfall. Our experiments demonstrate that incorporating randomness--particularly with the Gumbel-Softmax trick--can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking, resulting in superior performance across eight reasoning benchmarks. We further demonstrate that Stochastic Soft Thinking exhibits stronger exploration potential compared to conventional COT. Our findings deepen the understanding of continuous reasoning and establish the foundation for future work on improving Soft Thinking with Reinforcement Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14207v1">Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v1">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v2">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigate to what extent large language models (LLMs) can generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15191v1">Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: https://github.com/jlwu002/sr1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15188v1">OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15152v1">Tail-Optimized Caching for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Prompt caching is critical for reducing latency and cost in LLM inference: OpenAI and Anthropic report up to 50-90% cost savings through prompt reuse. Despite its widespread success, little is known about what constitutes an optimal prompt caching policy, particularly when optimizing tail latency, a metric of central importance to practitioners. The widely used Least Recently Used (LRU) policy can perform arbitrarily poor on this metric, as it is oblivious to the heterogeneity of conversation lengths. To address this gap, we propose Tail-Optimized LRU, a simple two-line modification that reallocates KV cache capacity to prioritize high-latency conversations by evicting cache entries that are unlikely to affect future turns. Though the implementation is simple, we prove its optimality under a natural stochastic model of conversation dynamics, providing the first theoretical justification for LRU in this setting, a result that may be of independent interest to the caching community. Experimentally, on real conversation data WildChat, Tail-Optimized LRU achieves up to 27.5% reduction in P90 tail Time to First Token latency and 23.9% in P95 tail latency compared to LRU, along with up to 38.9% decrease in SLO violations of 200ms. We believe this provides a practical and theoretically grounded option for practitioners seeking to optimize tail latency in real-world LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15144v1">HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
    </div>
    <details class="paper-abstract">
      Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15125v1">Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Under-submission
    </div>
    <details class="paper-abstract">
      Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically generating an interpretable topic taxonomy from an unlabeled corpus. By combining unsupervised clustering with prompt-based labeling, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets or domain expertise. We apply this framework to a large corpus of Meta (previously known as Facebook) political ads from the month ahead of the 2024 U.S. Presidential election. Our approach uncovers latent discourse structures, synthesizes semantically rich topic labels, and annotates topics with moral framing dimensions. We show quantitative and qualitative analyses to demonstrate the effectiveness of our framework. Our findings reveal that voting and immigration ads dominate overall spending and impressions, while abortion and election-integrity achieve disproportionate reach. Funding patterns are equally polarized: economic appeals are driven mainly by conservative PACs, abortion messaging splits between pro- and anti-rights coalitions, and crime-and-justice campaigns are fragmented across local committees. The framing of these appeals also diverges--abortion ads emphasize liberty/oppression rhetoric, while economic messaging blends care/harm, fairness/cheating, and liberty/oppression narratives. Topic salience further reveals strong correlations between moral foundations and issues. Demographic targeting also emerges. This work supports scalable, interpretable analysis of political messaging on social media, enabling researchers, policymakers, and the public to better understand emerging narratives, polarization dynamics, and the moral underpinnings of digital political communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15112v1">AndroByte: LLM-Driven Privacy Analysis through Bytecode Summarization and Dynamic Dataflow Call Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      With the exponential growth in mobile applications, protecting user privacy has become even more crucial. Android applications are often known for collecting, storing, and sharing sensitive user information such as contacts, location, camera, and microphone data often without the user's clear consent or awareness raising significant privacy risks and exposure. In the context of privacy assessment, dataflow analysis is particularly valuable for identifying data usage and potential leaks. Traditionally, this type of analysis has relied on formal methods, heuristics, and rule-based matching. However, these techniques are often complex to implement and prone to errors, such as taint explosion for large programs. Moreover, most existing Android dataflow analysis methods depend heavily on predefined list of sinks, limiting their flexibility and scalability. To address the limitations of these existing techniques, we propose AndroByte, an AI-driven privacy analysis tool that leverages LLM reasoning on bytecode summarization to dynamically generate accurate and explainable dataflow call graphs from static code analysis. AndroByte achieves a significant F\b{eta}-Score of 89% in generating dynamic dataflow call graphs on the fly, outperforming the effectiveness of traditional tools like FlowDroid and Amandroid in leak detection without relying on predefined propagation rules or sink lists. Moreover, AndroByte's iterative bytecode summarization provides comprehensive and explainable insights into dataflow and leak detection, achieving high, quantifiable scores based on the G-Eval metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15081v1">A Generalizable Rhetorical Strategy Annotation Model Using LLM-based Debate Simulation and Labelling</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Rhetorical strategies are central to persuasive communication, from political discourse and marketing to legal argumentation. However, analysis of rhetorical strategies has been limited by reliance on human annotation, which is costly, inconsistent, difficult to scale. Their associated datasets are often limited to specific topics and strategies, posing challenges for robust model development. We propose a novel framework that leverages large language models (LLMs) to automatically generate and label synthetic debate data based on a four-part rhetorical typology (causal, empirical, emotional, moral). We fine-tune transformer-based classifiers on this LLM-labeled dataset and validate its performance against human-labeled data on this dataset and on multiple external corpora. Our model achieves high performance and strong generalization across topical domains. We illustrate two applications with the fine-tuned model: (1) the improvement in persuasiveness prediction from incorporating rhetorical strategy labels, and (2) analyzing temporal and partisan shifts in rhetorical strategies in U.S. Presidential debates (1960-2020), revealing increased use of affective over cognitive argument in U.S. Presidential debates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v3">To Err Is Human; To Annotate, SILICON? Reducing Measurement Error in LLM Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation is foundational to management research and Large Language Models (LLMs) promise a cost-effective and scalable alternative to human annotation. The validity of insights drawn from LLM annotated data critically depends on minimizing the discrepancy between LLM assigned labels and the unobserved ground truth, as well as ensuring long-term reproducibility of results. We address the gap in the literature on LLM annotation by decomposing measurement error in LLM-based text annotation into four distinct sources: (1) guideline-induced error from inconsistent annotation criteria, (2) baseline-induced error from unreliable human reference standards, (3) prompt-induced error from suboptimal meta-instruction formatting, and (4) model-induced error from architectural differences across LLMs. We develop the SILICON methodology to systematically reduce measurement error from LLM annotation in all four sources above. Empirical validation across seven management research cases shows iteratively refined guidelines substantially increases the LLM-human agreement compared to one-shot guidelines; expert-generated baselines exhibit higher inter-annotator agreement as well as are less prone to producing misleading LLM-human agreement estimates compared to crowdsourced baselines; placing content in the system prompt reduces prompt-induced error; and model performance varies substantially across tasks. To further reduce error, we introduce a cost-effective multi-LLM labeling method, where only low-confidence items receive additional labels from alternative models. Finally, in addressing closed source model retirement cycles, we introduce an intuitive regression-based methodology to establish robust reproducibility protocols. Our evidence indicates that reducing each error source is necessary, and that SILICON supports reproducible, rigorous annotation in management research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v4">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at ACM International Conference on AI in Finance (ICAIF)
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15017v1">Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14973v1">Attention Is All You Need for KV Cache in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 https://vila-lab.github.io/elastic-cache-webpage/
    </div>
    <details class="paper-abstract">
      This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14972v1">TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14967v1">Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14966v1">Identity-Link IRT for Label-Free LLM Evaluation: Preserving Additivity in TVD-MI Scores</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Pairwise comparisons of large language models using total variation distance mutual information (TVD-MI) produce binary critic decisions per pair. We show that averaging TVD-MI's binary trials yields centered-probability scores with additive structure suitable for item-response theory (IRT) without nonlinear link functions. Maximum-likelihood approaches to IRT use logistic links, but we find empirically that these transformations introduce curvature that breaks additivity: across three domains, the identity link yields median curl on raw data of 0.080-0.150 (P95 = [0.474, 0.580]), whereas probit/logit introduce substantially higher violations (median [0.245, 0.588], P95 [0.825, 2.252]). We derive this clipped-linear model from Gini entropy maximization, yielding a box-constrained least-squares formulation that handles boundary saturation. At 33% coverage, we achieve holdout RMSE $0.117 \pm 0.008$ while preserving agent rankings (Spearman $\rho = 0.972 \pm 0.015$), three times fewer evaluations than full dense. Judge robustness analysis (GPT-4o-mini vs. Llama3-70b) shows strong agreement in agent rankings ($\rho = 0.872$) and consistent identity-link advantage. TVD-MI's geometry is best preserved by identity mapping for efficient LLM evaluation, applicable to other bounded-response domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14944v1">MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 22 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23339v2">VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6 pages, 1 figure, 1 algorithm, 5 tables, to be published in ISPACS 2025, unabridged version exists as arXiv:2506.23339v1
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate substantial promise for advancing scientific discovery, yet their deployment in disciplines demanding factual precision and specialized domain constraints presents significant challenges. Within molecular design for pharmaceutical development, these models can propose innovative molecular modifications but frequently generate chemically infeasible structures. We introduce VALID-Mol, a comprehensive framework that integrates chemical validation with LLM-driven molecular design, achieving an improvement in valid chemical structure generation from 3% to 83%. Our methodology synthesizes systematic prompt optimization, automated chemical verification, and domain-adapted fine-tuning to ensure dependable generation of synthesizable molecules with enhanced properties. Our contribution extends beyond implementation details to provide a transferable methodology for scientifically-constrained LLM applications with measurable reliability enhancements. Computational analyses indicate our framework generates promising synthesis candidates with up to 17-fold predicted improvements in target binding affinity while preserving synthetic feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13091v2">Unmasking Hiring Bias: Platform Data Analysis and Controlled Experiments on Bias in Online Freelance Marketplaces via RAG-LLM Generated Contents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Online freelance marketplaces, a rapidly growing part of the global labor market, are creating a fair environment where professional skills are the main factor for hiring. While these platforms can reduce bias from traditional hiring, the personal information in user profiles raises concerns about ongoing discrimination. Past studies on this topic have mostly used existing data, which makes it hard to control for other factors and clearly see the effect of things like gender or race. To solve these problems, this paper presents a new method that uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to create realistic, artificial freelancer profiles for controlled experiments. This approach effectively separates individual factors, enabling a clearer statistical analysis of how different variables influence the freelancer project process. In addition to analyzing extracted data with traditional statistical methods for post-project stage analysis, our research utilizes a dataset with highly controlled variables, generated by an RAG-LLM, to conduct a simulated hiring experiment for pre-project stage analysis. The results of our experiments show that, regarding gender, while no significant preference emerged in initial hiring decisions, female freelancers are substantially more likely to receive imperfect ratings post-project stage. Regarding regional bias, a strong and consistent preference favoring US-based freelancers shows that people are more likely to be selected in the simulated experiments, perceived as more leader-like, and receive higher ratings on the live platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v2">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach suffers from catastrophic forgetting: second-stage RL gradually loses SFT-acquired behaviors and inefficiently explores new patterns. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v3">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v1">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14824v1">Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      In information retrieval, training reranking models mainly focuses on two types of objectives: metric learning (e.g. contrastive loss to increase the predicted scores on relevant query-document pairs) and classification (binary label prediction of relevance vs. irrelevance). For BERT-style encoders, various studies have shown that contrastive learning (CL) can be more effective than discriminative (classification) learning. However, for large language models (LLMs), classification via supervised fine-tuning (SFT), which predicts ''yes'' (resp. ''no'') token for relevant (resp. irrelevant) pairs, appears more promising as it aligns well with the generative nature of LLMs. This divergence raises a central question: which objective is intrinsically better suited to LLM-based reranking, and what mechanism underlies the difference? In this work, we conduct a comprehensive comparison and analysis between CL and SFT for reranking, taking the universal multimodal retrieval (UMR) as the experimental playground. We first decompose the objectives into two components: weight, which controls the magnitude of those updates, and direction, which guides the model updates, then present a unified framework for understanding their interactions. Through probing experiments, we find that SFT provides a substantially stronger weighting scheme than CL, whereas the preferred scoring direction shows no clear winner. Taken together, these results point to a consistent advantage of SFT over CL for LLM reranking. To further validate our findings, we conduct large-scale training with SFT and present new state-of-the-art rerankers on the MRB benchmark. We also provide ablations on SFT settings and expect our findings to benefit future research and applications in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20921v2">LLM-guided Chemical Process Optimization with a Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages (main manuscript without references), 4 figures
    </div>
    <details class="paper-abstract">
      Chemical process optimization maximizes production efficiency and economic performance, but optimization algorithms, including gradient-based solvers, numerical methods, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable. We present a multi-agent LLM framework that autonomously infers operating constraints from minimal process descriptions, then collaboratively guides optimization. Our AutoGen-based framework employs OpenAI's o3 model with specialized agents for constraint generation, parameter validation, simulation, and optimization guidance. Through autonomous constraint generation and iterative multi-agent optimization, the framework eliminates the need for predefined operational bounds. Validated on hydrodealkylation across cost, yield, and yield-to-cost ratio metrics, the framework achieved competitive performance with conventional methods while reducing wall-time 31-fold relative to grid search, converging in under 20 minutes. The reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs and applying domain-informed heuristics. Unlike conventional methods requiring predefined constraints, our approach uniquely combines autonomous constraint generation with interpretable parameter exploration. Model comparison reveals reasoning-capable architectures (o3, o1) are essential for successful optimization, while standard models fail to converge. This approach is particularly valuable for emerging processes and retrofit applications where operational constraints are poorly characterized or unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20934v2">Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at the International Conference on Software Maintenance and Evolution (ICSME'25)
    </div>
    <details class="paper-abstract">
      MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07887v2">Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14756v1">Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14751v1">Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v3">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16191v2">The Last Dependency Crusade: Solving Python Dependency Conflicts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Pre-print - Accepted at the first annual workshop on Agentic Software Engineering (AgenticSE) co-located with ASE'25
    </div>
    <details class="paper-abstract">
      Resolving Python dependency issues remains a tedious and error-prone process, forcing developers to manually trial compatible module versions and interpreter configurations. Existing automated solutions, such as knowledge-graph-based and database-driven methods, face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This paper investigates the use of Large Language Models (LLMs) to automatically repair dependency issues in Python programs. We propose PLLM (pronounced "plum"), a novel retrieval-augmented generation (RAG) approach that iteratively infers missing or incorrect dependencies. PLLM builds a test environment where the LLM proposes module combinations, observes execution feedback, and refines its predictions using natural language processing (NLP) to parse error messages. We evaluate PLLM on the Gistable HG2.9K dataset, a curated collection of real-world Python programs. Using this benchmark, we explore multiple PLLM configurations, including six open-source LLMs evaluated both with and without RAG. Our findings show that RAG consistently improves fix rates, with the best performance achieved by Gemma-2 9B when combined with RAG. Compared to two state-of-the-art baselines, PyEGo and ReadPyE, PLLM achieves significantly higher fix rates; +15.97\% more than ReadPyE and +21.58\% more than PyEGo. Further analysis shows that PLLM is especially effective for projects with numerous dependencies and those using specialized numerical or machine-learning libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14700v1">LLM Agents for Automated Web Vulnerability Reproduction: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in software engineering and cybersecurity tasks, including code generation, vulnerability discovery, and automated testing. One critical but underexplored application is automated web vulnerability reproduction, which transforms vulnerability reports into working exploits. Although recent advances suggest promising potential, challenges remain in applying LLM agents to real-world web vulnerability reproduction scenarios. In this paper, we present the first comprehensive evaluation of state-of-the-art LLM agents for automated web vulnerability reproduction. We systematically assess 20 agents from software engineering, cybersecurity, and general domains across 16 dimensions, including technical capabilities, environment adaptability, and user experience factors, on 3 representative web vulnerabilities. Based on the results, we select three top-performing agents (OpenHands, SWE-agent, and CAI) for in-depth evaluation on our benchmark dataset of 80 real-world CVEs spanning 7 vulnerability types and 6 web technologies. Our results reveal that while LLM agents achieve reasonable success on simple library-based vulnerabilities, they consistently fail on complex service-based vulnerabilities requiring multi-component environments. Complex environment configurations and authentication barriers create a gap where agents can execute exploit code but fail to trigger actual vulnerabilities. We observe high sensitivity to input guidance, with performance degrading by over 33% under incomplete authentication information. Our findings highlight the significant gap between current LLM agent capabilities and the demands of reliable automated vulnerability reproduction, emphasizing the need for advances in environmental adaptation and autonomous problem-solving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13925v1">An LLM-Powered AI Agent Framework for Holistic IoT Traffic Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Internet of Things (IoT) networks generate diverse and high-volume traffic that reflects both normal activity and potential threats. Deriving meaningful insight from such telemetry requires cross-layer interpretation of behaviors, protocols, and context rather than isolated detection. This work presents an LLM-powered AI agent framework that converts raw packet captures into structured and semantically enriched representations for interactive analysis. The framework integrates feature extraction, transformer-based anomaly detection, packet and flow summarization, threat intelligence enrichment, and retrieval-augmented question answering. An AI agent guided by a large language model performs reasoning over the indexed traffic artifacts, assembling evidence to produce accurate and human-readable interpretations. Experimental evaluation on multiple IoT captures and six open models shows that hybrid retrieval, which combines lexical and semantic search with reranking, substantially improves BLEU, ROUGE, METEOR, and BERTScore results compared with dense-only retrieval. System profiling further indicates low CPU, GPU, and memory overhead, demonstrating that the framework achieves holistic and efficient interpretation of IoT network traffic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13918v1">Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Process reward models (PRMs) are a cornerstone of test-time scaling (TTS), designed to verify and select the best responses from large language models (LLMs). However, this promise is challenged by recent benchmarks where simple majority voting, which ignores PRM signals, occasionally outperforms standard PRM-based selection. This raises a critical question: How can we effectively utilize verification signals from PRMs for TTS? To address this, we start by developing a theoretical framework for optimally combining signals from both the LLM and the PRM. Our framework reveals that the optimal strategy is a weighted aggregation of responses, a strategy whose effectiveness hinges on estimating weights that capture the complex interplay between the models. Based on our theoretical results, we empirically show that these optimal weighting functions differ significantly across LLM-PRM pairs and, notably, often assign substantial negative weights. Motivated by these insights, we propose efficient pre-computation methods to calibrate these weighting functions. Extensive experiments across 5 LLMs and 7 PRMs demonstrate that our calibration method significantly boosts the TTS efficiency, surpassing the performance of vanilla weighted majority voting while using only $21.3\%$ of the computation. Ultimately, our work demonstrates that investing in a more intelligent aggregation strategy can be a more convincing path to performance gains than simply scaling test-time computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13914v1">A11YN: aligning LLMs for accessible web UI code generation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong capabilities in generating functional and aesthetic web interfaces directly from instructions. However, these models often replicate accessibility flaws from their training data, resulting in interfaces that exclude users with diverse needs and contexts. To address this gap, we introduce A11yn, the first method that aligns code-generating LLMs to reliably produce accessibility-compliant web UIs. A11yn optimizes a novel reward function that penalizes violations of the Web Content Accessibility Guidelines (WCAG), with penalties scaled to the severity of each violation as identified by an accessibility testing engine. To support training, we construct UIReq-6.8K, a dataset of 6,800 diverse instructions for web UI generation. For evaluation, we introduce RealUIReq-300, a benchmark of 300 real-world web UI requests grounded and manually curated from public web pages, spanning a broad range of use cases. Empirical results show that A11yn significantly outperforms strong baselines, lowering the Inaccessibility Rate by 60% over the base model while preserving semantic fidelity and visual quality of generated UIs. These findings demonstrate that accessibility can be systematically optimized within LLMs, showing the feasibility of aligning code generation for accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13910v1">RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14896v2">Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Technical Report, Work in Progress
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. Our code is publicly available at https://github.com/FelixMessi/QDLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14304v2">Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13786v1">The Art of Scaling Reinforcement Learning Compute for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 28 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to training large language models (LLMs), yet the field lacks predictive scaling methodologies comparable to those established for pre-training. Despite rapidly rising compute budgets, there is no principled understanding of how to evaluate algorithmic improvements for scaling RL compute. We present the first large-scale systematic study, amounting to more than 400,000 GPU-hours, that defines a principled framework for analyzing and predicting RL scaling in LLMs. We fit sigmoidal compute-performance curves for RL training and ablate a wide range of common design choices to analyze their effects on asymptotic performance and compute efficiency. We observe: (1) Not all recipes yield similar asymptotic performance, (2) Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote, and (3) Stable, scalable recipes follow predictable scaling trajectories, enabling extrapolation from smaller-scale runs. Combining these insights, we propose a best-practice recipe, ScaleRL, and demonstrate its effectiveness by successfully scaling and predicting validation performance on a single RL run scaled up to 100,000 GPU-hours. Our work provides both a scientific framework for analyzing scaling in RL and a practical recipe that brings RL training closer to the predictability long achieved in pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12470v3">Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19831v2">Benchmarking Hindi LLMs: A New Suite of Datasets and a Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Evaluating instruction-tuned Large Language Models (LLMs) in Hindi is challenging due to a lack of high-quality benchmarks, as direct translation of English datasets fails to capture crucial linguistic and cultural nuances. To address this, we introduce a suite of five Hindi LLM evaluation datasets: IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi. These were created using a methodology that combines from-scratch human annotation with a translate-and-verify process. We leverage this suite to conduct an extensive benchmarking of open-source LLMs supporting Hindi, providing a detailed comparative analysis of their current capabilities. Our curation process also serves as a replicable methodology for developing benchmarks in other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13750v1">Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 UncertaiNLP at EMNLP 2025
    </div>
    <details class="paper-abstract">
      We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13738v1">HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong potential for sequential recommendation. However, current LLM-based approaches face critical limitations in modeling users' long-term and diverse interests. First, due to inference latency and feature fetching bandwidth constraints, existing methods typically truncate user behavior sequences to include only the most recent interactions, resulting in the loss of valuable long-range preference signals. Second, most current methods rely on next-item prediction with a single predicted embedding, overlooking the multifaceted nature of user interests and limiting recommendation diversity. To address these challenges, we propose HyMiRec, a hybrid multi-interest sequential recommendation framework, which leverages a lightweight recommender to extracts coarse interest embeddings from long user sequences and an LLM-based recommender to captures refined interest embeddings. To alleviate the overhead of fetching features, we introduce a residual codebook based on cosine similarity, enabling efficient compression and reuse of user history embeddings. To model the diverse preferences of users, we design a disentangled multi-interest learning module, which leverages multiple interest queries to learn disentangles multiple interest signals adaptively, allowing the model to capture different facets of user intent. Extensive experiments are conducted on both benchmark datasets and a collected industrial dataset, demonstrating our effectiveness over existing state-of-the-art methods. Furthermore, online A/B testing shows that HyMiRec brings consistent improvements in real-world recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13713v1">Don't Be Greedy, Just Relax! Pruning LLMs via Frank-Wolfe</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Pruning is a common technique to reduce the compute and storage requirements of Neural Networks. While conventional approaches typically retrain the model to recover pruning-induced performance degradation, state-of-the-art Large Language Model (LLM) pruning methods operate layer-wise, minimizing the per-layer pruning error on a small calibration dataset to avoid full retraining, which is considered computationally prohibitive for LLMs. However, finding the optimal pruning mask is a hard combinatorial problem and solving it to optimality is intractable. Existing methods hence rely on greedy heuristics that ignore the weight interactions in the pruning objective. In this work, we instead consider the convex relaxation of these combinatorial constraints and solve the resulting problem using the Frank-Wolfe (FW) algorithm. Our method drastically reduces the per-layer pruning error, outperforms strong baselines on state-of-the-art GPT architectures, and remains memory-efficient. We provide theoretical justification by showing that, combined with the convergence guarantees of the FW algorithm, we obtain an approximate solution to the original combinatorial problem upon rounding the relaxed solution to integrality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08615v2">Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Mathematical reasoning serves as a crucial testbed for evaluating the intelligence of large language models (LLMs), and math word problems (MWPs) represent one of the most widely used formats. Most existing MWP datasets contain only the necessary information, while problems with distracting or excessive conditions are often overlooked. Prior studies have shown that popular LLMs experience a dramatic performance drop when such distracting conditions are introduced. However, available datasets of MWPs with distracting conditions remain limited, and most exhibit low difficulty and out-of-context expressions. These shortcomings make the distracting conditions easy to detect and disregard, thereby reducing the credibility of benchmarking on these datasets. Moreover, when distracting conditions are added, the reasoning process and answers may change, requiring intensive manual effort to check and rewrite solutions. To address these issues, we design an iterative framework that leverages LLMs to generate distracting conditions automatically. We develop a set of prompts to revise MWPs from multiple perspectives and cognitive levels, encouraging the creation of meaningful distracting conditions as well as suggestions for further refinement. A key advantage of our framework is the preservation of shared solutions between the original and revised problems: the LLMs are explicitly guided to generate distractions that do not alter the original solution, thus eliminating the need to produce new answers. This framework is efficient and easy to deploy, substantially reducing the effort required to generate MWPs with distracting conditions while maintaining high data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14556v2">LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are increasingly being utilized in various private and commercial applications, e.g., traffic control, parcel delivery, and Search and Rescue (SAR) missions. Machine Learning (ML) methods used in UAV-Assisted Sensor Networks (UASNETs) and, especially, in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sampling efficiency, which conflict with the urgency of emergencies, such as SAR missions. In this paper, an In-Context Learning (ICL)-Data Collection Scheduling (ICLDC) system is proposed as an alternative to DRL in emergencies. The UAV collects sensory data and transmits it to a Large Language Model (LLM), which creates a task description in natural language. From this description, the UAV receives a data collection schedule that must be executed. A verifier ensures safe UAV operations by evaluating the schedules generated by the LLM and overriding unsafe schedules based on predefined rules. The system continuously adapts by incorporating feedback into the task descriptions and using this for future decisions. This method is tested against jailbreaking attacks, where the task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC significantly reduces cumulative packet loss compared to both the DQN and Maximum Channel Gain baselines. ICLDC presents a promising direction for intelligent scheduling and control in UASNETs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13668v1">Adaptive Rescheduling in Prefill-Decode Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference has emerged as a fundamental paradigm. In real-world scenarios, variations in output length cause severe workload imbalance in the decode phase, particularly for long-output reasoning tasks. Existing systems, such as PD disaggregation architectures, rely on static prefill-to-decode scheduling, which often results in SLO violations and OOM failures under evolving decode workloads. In this paper, we propose ARES, an adaptive decoding rescheduling system powered by length prediction to anticipate future workloads. Our core contributions include: (1) A lightweight and continuous LLM-native prediction method that leverages LLM hidden state to model remaining generation length with high precision (reducing MAE by 49.42%) and low overhead (cutting predictor parameters by 93.28%); (2) A rescheduling solution in decode phase with : A dynamic balancing mechanism that integrates current and predicted workloads, reducing P99 TPOT by 74.77% and achieving up to 2.24 times higher goodput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19234v2">GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration faces critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization. The code is available at https://github.com/JialongZhou666/GUARDIAN
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13632v1">Closing the Gap Between Text and Speech Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13624v1">Unlocking Public Catalogues: Instruction-Tuning LLMs for ICD Coding of German Tumor Diagnoses</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 19 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM accuracy rose from 1.4-24% to 41-58%, and partial accuracy from 31-74% to 73-83%. The accuracy of ICD-O-3 topography coding also improved but started and remained considerably lower with an exact accuracy of 22-40% and a partial accuracy of 56-67% after fine-tuning. Malformed code outputs dropped to 0% for all models. Tumor-diagnosis recognition reached 99%. Accuracy correlated positively with model size, but gaps between small and large models narrowed after fine-tuning. The reasoning mode in Qwen3 generally yielded a lower performance than fine-tuning and was over 100 times slower. Our findings highlight the potential of leveraging public catalogues to build instruction datasets that improve LLMs in medical documentation tasks. The complete training dataset and the best-performing checkpoints of the fine-tuned models are available from https://huggingface.co/datasets/stefan-m-lenz/ICDOPS-QA-2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v3">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted at ACM International Conference on AI in Finance (ICAIF)
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13586v1">Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13575v1">Auto-repair without test cases: How LLMs fix compilation errors in large industrial embedded code</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 9 pages, 4 figures, conference: 2025 28th Euromicro Conference on Digital System Design (DSD)
    </div>
    <details class="paper-abstract">
      The co-development of hardware and software in industrial embedded systems frequently leads to compilation errors during continuous integration (CI). Automated repair of such failures is promising, but existing techniques rely on test cases, which are not available for non-compilable code. We employ an automated repair approach for compilation errors driven by large language models (LLMs). Our study encompasses the collection of more than 40000 commits from the product's source code. We assess the performance of an industrial CI system enhanced by four state-of-the-art LLMs, comparing their outcomes with manual corrections provided by human programmers. LLM-equipped CI systems can resolve up to 63 % of the compilation errors in our baseline dataset. Among the fixes associated with successful CI builds, 83 % are deemed reasonable. Moreover, LLMs significantly reduce debugging time, with the majority of successful cases completed within 8 minutes, compared to hours typically required for manual debugging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13570v1">Selective Adversarial Attacks on LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Benchmarking outcomes increasingly govern trust, selection, and deployment of LLMs, yet these evaluations remain vulnerable to semantically equivalent adversarial perturbations. Prior work on adversarial robustness in NLP has emphasized text attacks that affect many models equally, leaving open the question of whether it is possible to selectively degrade or enhance performance while minimally affecting other models. We formalize this problem and study selective adversarial attacks on MMLU - a widely used benchmark designed to measure a language model's broad general knowledge and reasoning ability across different subjects. Using canonical attacks integrated into TextAttack framework, we introduce a protocol for selectivity assessment, develop a custom constraint to increase selectivity of attacks and propose a surrogate-LLM pipeline that generates selective perturbations. Empirically, we find that selective adversarial attacks exist and can materially alter relative rankings, challenging the fairness, reproducibility, and transparency of leaderboard-driven evaluation. Our results motivate perturbation-aware reporting and robustness diagnostics for LLM evaluation and demonstrate that even subtle edits can shift comparative judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13554v1">Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 23 pages, 8 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The reasoning pattern of Large language models (LLMs) remains opaque, and Reinforcement learning (RL) typically applies uniform credit across an entire generation, blurring the distinction between pivotal and routine steps. This work positions attention as a privileged substrate that renders the internal logic of LLMs legible, not merely as a byproduct of computation, but as a mechanistic blueprint of reasoning itself. We first distinguish attention heads between locally and globally focused information processing and reveal that locally focused heads produce a sawtooth pattern near the diagonal indicating phrasal chunks, while globally focused heads expose tokens that exert broad downstream influence over future tokens. We formalize these with two metrics: 1) Windowed Average Attention Distance, which measures the extent of backward attention within a clipped window; 2) Future Attention Influence, which quantifies a token's global importance as the average attention it receives from subsequent tokens. Taken together, these signals reveal a recurring preplan-and-anchor mechanism, where the model first performs a long-range contextual reference to generate an introductory token, which is immediately followed by or coincides with a semantic anchor token that organizes subsequent reasoning. Leveraging these insights, we introduce three novel RL strategies that dynamically perform targeted credit assignment to critical nodes (preplan tokens, anchor tokens, and their temporal coupling) and show consistent performance gains across various reasoning tasks. By aligning optimization with the model's intrinsic reasoning rhythm, we aim to transform opaque optimization into an actionable structure-aware process, hoping to offer a potential step toward more transparent and effective optimization of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13543v1">In-Browser LLM-Guided Fuzzing for Real-Time Prompt Injection Testing in Agentic AI Browsers</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 37 pages , 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based agents integrated into web browsers (often called agentic AI browsers) offer powerful automation of web tasks. However, they are vulnerable to indirect prompt injection attacks, where malicious instructions hidden in a webpage deceive the agent into unwanted actions. These attacks can bypass traditional web security boundaries, as the AI agent operates with the user privileges across sites. In this paper, we present a novel fuzzing framework that runs entirely in the browser and is guided by an LLM to automatically discover such prompt injection vulnerabilities in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04364v4">Benchmarking LLMs' Swarm intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13501v1">Confidence as a Reward: Transforming LLMs into Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Reward models can significantly enhance the reasoning capabilities of large language models (LLMs), but they typically require extensive curated data and costly training. To mitigate these challenges, training-free approaches such as LLM-as-a-Judge leverage the intrinsic reasoning abilities of LLMs to evaluate responses, achieving promising results. Recent works have also indicated that model confidence can serve effectively as a reward metric, distinguishing between chain-of-thought (CoT) and non-CoT paths. However, the concept of using confidence as a reward has not been comprehensively studied. In this work, we systematically investigate Confidence-as-a-Reward (CRew), a simple yet powerful training-free method that utilizes token-level confidence in the model's final answers as a proxy for reward, especially suitable for close-ended tasks. Through extensive experiments on mathematical reasoning tasks, we demonstrate that CRew outperforms existing training-free reward approaches on the MATH500 and RewardMATH benchmarks, and even surpasses most trained reward models. We further identify a strong correlation between CRew scores and the actual reasoning performance of the model. Additionally, we find that CRew can effectively filter high-quality training data. Building upon these insights, we propose CRew-DPO, a training strategy that constructs preference data from confidence scores combined with correctness signals. Finetuning with CRew-DPO further enhances the model's judging capabilities and consistently outperforms existing self-training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13500v1">MedREK: Retrieval-Based Editing for Medical LLMs with Key-Aware Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      LLMs hold great promise for healthcare applications, but the rapid evolution of medical knowledge and errors in training data often cause them to generate outdated or inaccurate information, limiting their applicability in high-stakes clinical practice. Model editing has emerged as a potential remedy without full retraining. While parameter-based editing often compromises locality and is thus ill-suited for the medical domain, retrieval-based editing offers a more viable alternative. However, it still faces two critical challenges: (1) representation overlap within the medical knowledge space often causes inaccurate retrieval and reduces editing accuracy; (2) existing methods are restricted to single-sample edits, while batch-editing remains largely unexplored despite its importance for real-world medical applications. To address these challenges, we first construct MedVersa, \hk{an enhanced benchmark with broader coverage of medical subjects, designed to evaluate both single and batch edits under strict locality constraints}. We then propose MedREK, a retrieval-based editing framework that integrates a shared query-key module for precise matching with an attention-based prompt encoder for informative guidance. Experimental results on various medical benchmarks demonstrate that our MedREK achieves superior performance across different core metrics and provides the first validated solution for batch-editing in medical LLMs. Our code and dataset are available at https://github.com/mylittleriver/MedREK.
    </details>
</div>
