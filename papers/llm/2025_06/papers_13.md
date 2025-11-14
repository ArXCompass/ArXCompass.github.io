# llm - 2025_06

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
- Part 13
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01821v2">On the Power of Context-Enhanced Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 77 pages, 18 figures; ICML 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We formalize a new concept for LLMs, context-enhanced learning. It involves standard gradient-based learning on text except that the context is enhanced with additional data on which no auto-regressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works. Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04482v1">Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Findings of the Association for Computational Linguistics: ACL 2025
    </div>
    <details class="paper-abstract">
      The NLP research community has made publicly available numerous instruments for measuring representational harms caused by large language model (LLM)-based systems. These instruments have taken the form of datasets, metrics, tools, and more. In this paper, we examine the extent to which such instruments meet the needs of practitioners tasked with evaluating LLM-based systems. Via semi-structured interviews with 12 such practitioners, we find that practitioners are often unable to use publicly available instruments for measuring representational harms. We identify two types of challenges. In some cases, instruments are not useful because they do not meaningfully measure what practitioners seek to measure or are otherwise misaligned with practitioner needs. In other cases, instruments - even useful instruments - are not used by practitioners due to practical and institutional barriers impeding their uptake. Drawing on measurement theory and pragmatic measurement, we provide recommendations for addressing these challenges to better meet practitioner needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04481v1">CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) show promise in solving complex mathematical tasks, existing evaluation paradigms rely solely on a coarse measure of overall answer accuracy, which are insufficient for assessing their authentic capabilities. In this paper, we propose \textbf{CogMath}, which comprehensively assesses LLMs' mathematical abilities through the lens of human cognition. Specifically, inspired by psychological theories, CogMath formalizes human reasoning process into 3 stages: \emph{problem comprehension}, \emph{problem solving}, and \emph{solution summarization}. Within these stages, we investigate perspectives such as numerical calculation, knowledge, and counterfactuals, and design a total of 9 fine-grained evaluation dimensions. In each dimension, we develop an ``\emph{Inquiry}-\emph{Judge}-\emph{Reference}'' multi-agent system to generate inquiries that assess LLMs' mastery from this dimension. An LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. By applying CogMath on three benchmarks, we reveal that the mathematical capabilities of 7 mainstream LLMs are overestimated by 30\%-40\%. Moreover, we locate their strengths and weaknesses across specific stages/dimensions, offering in-depth insights to further enhance their reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04478v1">Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v1">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 19 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v1">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Submitted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches this graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04418v1">Characterizing Multi-Hunk Patches: Divergence, Proximity, and LLM Repair Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Multi-hunk bugs, where fixes span disjoint regions of code, are common in practice, yet remain underrepresented in automated repair. Existing techniques and benchmarks pre-dominantly target single-hunk scenarios, overlooking the added complexity of coordinating semantically related changes across the codebase. In this work, we characterize HUNK4J, a dataset of multi-hunk patches derived from 372 real-world defects. We propose hunk divergence, a metric that quantifies the variation among edits in a patch by capturing lexical, structural, and file-level differences, while incorporating the number of hunks involved. We further define spatial proximity, a classification that models how hunks are spatially distributed across the program hierarchy. Our empirical study spanning six LLMs reveals that model success rates decline with increased divergence and spatial dispersion. Notably, when using the LLM alone, no model succeeds in the most dispersed Fragment class. These findings highlight a critical gap in LLM capabilities and motivate divergence-aware repair strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v3">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04405v1">MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v3">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by IJCAI 2025 (Survey Track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs, which can be categorized into the following two aspects: (1) Logical question answering: LLMs often fail to generate the correct answer within a complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency: LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art question-answering LLM Macaw, answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose a detailed taxonomy. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistencies, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extending to modal logic to account for uncertainty and developing efficient algorithms that simultaneously satisfy multiple logical consistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v3">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      In this work, we investigate the causal reasoning abilities of large language models (LLMs) through the representative problem of inferring causal relationships from narratives. We find that even state-of-the-art language models rely on unreliable shortcuts, both in terms of the narrative presentation and their parametric knowledge. For example, LLMs tend to determine causal relationships based on the topological ordering of events (i.e., earlier events cause later ones), resulting in lower performance whenever events are not narrated in their exact causal order. Similarly, we demonstrate that LLMs struggle with long-term causal reasoning and often fail when the narratives are long and contain many events. Additionally, we show LLMs appear to rely heavily on their parametric knowledge at the expense of reasoning over the provided narrative. This degrades their abilities whenever the narrative opposes parametric knowledge. We extensively validate these failure modes through carefully controlled synthetic experiments, as well as evaluations on real-world narratives. Finally, we observe that explicitly generating a causal graph generally improves performance while naive chain-of-thought is ineffective. Collectively, our results distill precise failure modes of current state-of-the-art models and can pave the way for future techniques to enhance causal reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v3">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Added benchmarks, baselines, author, appendix
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., "Sure, here it is") or otherwise unsafe content -- even for harmful prompts. In experiments on the SafeBench and MM-SafetyBench benchmarks, our method achieves higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 81% on certain models). We further demonstrate cross-model universality by training on several multimodal LLMs simultaneously. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04344v1">GEM: Empowering LLM for both Embedding Generation and Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large decoder-only language models (LLMs) have achieved remarkable success in generation and reasoning tasks, where they generate text responses given instructions. However, many applications, e.g., retrieval augmented generation (RAG), still rely on separate embedding models to generate text embeddings, which can complicate the system and introduce discrepancies in understanding of the query between the embedding model and LLMs. To address this limitation, we propose a simple self-supervised approach, Generative Embedding large language Model (GEM), that enables any large decoder-only LLM to generate high-quality text embeddings while maintaining its original text generation and reasoning capabilities. Our method inserts new special token(s) into a text body, and generates summarization embedding of the text by manipulating the attention mask. This method could be easily integrated into post-training or fine tuning stages of any existing LLMs. We demonstrate the effectiveness of our approach by applying it to two popular LLM families, ranging from 1B to 8B parameters, and evaluating the transformed models on both text embedding benchmarks (MTEB) and NLP benchmarks (MMLU). The results show that our proposed method significantly improves the original LLMs on MTEB while having a minimal impact on MMLU. Our strong results indicate that our approach can empower LLMs with state-of-the-art text embedding capabilities while maintaining their original NLP performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04290v1">Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 20 pages, 6 figures, preprint under review at Information Processing & Management
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM), which have developed in recent years, enable credit risk assessment through the analysis of financial texts such as analyst reports and corporate disclosures. This paper presents the first systematic review and taxonomy focusing on LLMbased approaches in credit risk estimation. We determined the basic model architectures by selecting 60 relevant papers published between 2020-2025 with the PRISMA research strategy. And we examined the data used for scenarios such as credit default prediction and risk analysis. Since the main focus of the paper is interpretability, we classify concepts such as explainability mechanisms, chain of thought prompts and natural language justifications for LLM-based credit models. The taxonomy organizes the literature under four main headings: model architectures, data types, explainability mechanisms and application areas. Based on this analysis, we highlight the main future trends and research gaps for LLM-based credit scoring systems. This paper aims to be a reference paper for artificial intelligence and financial researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04282v1">DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Symbolic regression is a fundamental tool for discovering interpretable mathematical expressions from data, with broad applications across scientific and engineering domains. Recently, large language models (LLMs) have demonstrated strong performance in this task, leveraging embedded scientific priors and reasoning capabilities to surpass traditional methods. However, existing LLM-based approaches, such as LLM-SR, often over-rely on internal priors, lacking explicit data understanding and systematic reflection during equation generation. To address these limitations, we propose DrSR (Dual Reasoning Symbolic Regression), a framework that combines data-driven insight with reflective learning to enhance both robustness and discovery capability. Specifically, DrSR guides LLMs to analyze structural relationships (e.g., monotonicity, nonlinearity, and correlation) within the data to generate structured descriptions. Simultaneously, it monitors equation performance and establishes a feedback loop to refine subsequent generations. By integrating data understanding and generation reflection in a closed loop, DrSR enables more efficient exploration of the symbolic expression space. Experiments across interdisciplinary datasets in physics, chemistry, biology, and materials science demonstrate that DrSR substantially improves the valid equation rate and consistently outperforms both classical and recent LLM-based methods in terms of accuracy, generalization, and search efficiency. These results underscore its potential for scientific equation discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05414v1">SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Project website with demo videos: https://zijuncui02.github.io/SAVVY/
    </div>
    <details class="paper-abstract">
      3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05413v1">SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 6 pages, 3 figures, 5 tables. Submitted to the IEEE SMC 2025 conference
    </div>
    <details class="paper-abstract">
      We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05410v1">Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 14 pages,7 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v1">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04185v1">R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have notably progressed in multi-step and long-chain reasoning. However, extending their reasoning capabilities to encompass deep interactions with search remains a non-trivial challenge, as models often fail to identify optimal reasoning-search interaction trajectories, resulting in suboptimal responses. We propose R-Search, a novel reinforcement learning framework for Reasoning-Search integration, designed to enable LLMs to autonomously execute multi-step reasoning with deep search interaction, and learn optimal reasoning search interaction trajectories via multi-reward signals, improving response quality in complex logic- and knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when to retrieve or reason, while globally integrating key evidence to enhance deep knowledge interaction between reasoning and search. During RL training, R-Search provides multi-stage, multi-type rewards to jointly optimize the reasoning-search trajectory. Experiments on seven datasets show that R-Search outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1% (out-of-domain). The code and data are available at https://github.com/QingFei1/R-Search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13865v2">A Survey on (M)LLM-Based GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04172v1">Does Prompt Design Impact Quality of Data Imputation by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Generating realistic synthetic tabular data presents a critical challenge in machine learning. It adds another layer of complexity when this data contain class imbalance problems. This paper presents a novel token-aware data imputation method that leverages the in-context learning capabilities of large language models. This is achieved through the combination of a structured group-wise CSV-style prompting technique and the elimination of irrelevant contextual information in the input prompt. We test this approach with two class-imbalanced binary classification datasets and evaluate the effectiveness of imputation using classification-based evaluation metrics. The experimental results demonstrate that our approach significantly reduces the input prompt size while maintaining or improving imputation quality compared to our baseline prompt, especially for datasets that are of relatively smaller in size. The contributions of this presented work is two-fold -- 1) it sheds light on the importance of prompt design when leveraging LLMs for synthetic data generation and 2) it addresses a critical gap in LLM-based data imputation for class-imbalanced datasets with missing data by providing a practical solution within computational constraints. We hope that our work will foster further research and discussions about leveraging the incredible potential of LLMs and prompt engineering techniques for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v2">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04167v1">Neural and Cognitive Impacts of AI: The Influence of Task Subjectivity on Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      AI-based interactive assistants are advancing human-augmenting technology, yet their effects on users' mental and physiological states remain under-explored. We address this gap by analyzing how Copilot for Microsoft Word, a LLM-based assistant, impacts users. Using tasks ranging from objective (SAT reading comprehension) to subjective (personal reflection), and with measurements including fNIRS, Empatica E4, NASA-TLX, and questionnaires, we measure Copilot's effects on users. We also evaluate users' performance with and without Copilot across tasks. In objective tasks, participants reported a reduction of workload and an increase in enjoyment, which was paired with objective performance increases. Participants reported reduced workload and increased enjoyment with no change in performance in a creative poetry writing task. However, no benefits due to Copilot use were reported in a highly subjective self-reflection task. Although no physiological changes were recorded due to Copilot use, task-dependent differences in prefrontal cortex activation offer complementary insights into the cognitive processes associated with successful and unsuccessful human-AI collaboration. These findings suggest that AI assistants' effectiveness varies with task type-particularly showing decreased usefulness in tasks that engage episodic memory-and presents a brain-network based hypothesis of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04142v1">Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) depends on trustworthy evaluation. However, most current evaluations rely on public benchmarks, which are prone to data contamination issues that significantly compromise fairness. Previous researches have focused on constructing dynamic benchmarks to address contamination. However, continuously building new benchmarks is costly and cyclical. In this work, we aim to tackle contamination by analyzing the mechanisms of contaminated models themselves. Through our experiments, we discover that the overestimation of contaminated models is likely due to parameters acquiring shortcut solutions in training. We further propose a novel method for identifying shortcut neurons through comparative and causal analysis. Building on this, we introduce an evaluation method called shortcut neuron patching to suppress shortcut neurons. Experiments validate the effectiveness of our approach in mitigating contamination. Additionally, our evaluation results exhibit a strong linear correlation with MixEval, a recently released trustworthy benchmark, achieving a Spearman coefficient ($\rho$) exceeding 0.95. This high correlation indicates that our method closely reveals true capabilities of the models and is trustworthy. We conduct further experiments to demonstrate the generalizability of our method across various benchmarks and hyperparameter settings. Code: https://github.com/GaryStack/Trustworthy-Evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10981v2">Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL 2025 Main, 33 pages, 51 figures
    </div>
    <details class="paper-abstract">
      Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a probabilistic method to efficiently predict scaling performance and identify the best prompting strategy under large sampling times, eliminating the need for resource-intensive inference processes in practical applications. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. Code is available at https://github.com/MraDonkey/rethinking_prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v1">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built on large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligent autonomy, collaboration and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based agentic multi-agent systems (AMAS). We begin by examining the conceptual foundations of agentic AI, its architectural differences from traditional AI agents, and the emerging system designs that enable scalable, tool-using autonomy. The TRiSM in the agentic AI framework is then detailed through four pillars governance, explainability, ModelOps, and privacy/security each contextualized for agentic LLMs. We identify unique threat vectors and introduce a comprehensive risk taxonomy for the agentic AI applications, supported by case studies illustrating real-world vulnerabilities. Furthermore, the paper also surveys trust-building mechanisms, transparency and oversight techniques, and state-of-the-art explainability strategies in distributed LLM agent systems. Additionally, metrics for evaluating trust, interpretability, and human-centered performance are reviewed alongside open benchmarking challenges. Security and privacy are addressed through encryption, adversarial defense, and compliance with evolving AI regulations. The paper concludes with a roadmap for responsible agentic AI, proposing research directions to align emerging multi-agent systems with robust TRiSM principles for safe, accountable, and transparent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19176v2">Assistant-Guided Mitigation of Teacher Preference Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge employs large language models (LLMs), such as GPT-4, to evaluate the quality of LLM-generated responses, gaining popularity for its cost-effectiveness and strong alignment with human evaluations. However, training proxy judge models using evaluation data generated by powerful teacher models introduces a critical yet previously overlooked issue: teacher preference bias, where the proxy judge model learns a biased preference for responses from the teacher model. To tackle this problem, we propose a novel setting that incorporates an additional assistant model, which is not biased toward the teacher model's responses, to complement the training data. Building on this setup, we introduce AGDe-Judge, a three-stage framework designed to debias from both the labels and feedbacks in the training data. Extensive experiments demonstrate that AGDe-Judge effectively reduces teacher preference bias while maintaining strong performance across six evaluation benchmarks. Code is available at https://github.com/Liuz233/AGDe-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17089v2">KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Inference for Large Language Models (LLMs) is computationally demanding. To reduce the cost of auto-regressive decoding, Key-Value (KV) cache is used to store intermediate activations, which significantly lowers the computational overhead for token generation. However, the memory required for the KV cache grows rapidly, often exceeding the capacity of GPU memory. A cost-effective alternative is to offload KV cache to CPU memory, which alleviates GPU memory pressure, but shifts the bottleneck to the limited bandwidth of the PCIe connection between the CPU and GPU. Existing methods attempt to address these issues by overlapping GPU computation with I/O or employing CPU-GPU heterogeneous execution, but they are hindered by excessive data movement and dependence on CPU capabilities. Fully overlapping PCIe communication latency gets challenging as the size of the KV cache grows and/or the GPU compute capabilities increase. In this paper, we introduce KVPR, an efficient I/O-aware LLM inference method where the CPU first transfers a partial set of activations, from which the GPU can start recomputing the KV cache values. While the GPU recomputes the partial KV cache, the remaining portion of the KV cache is transferred concurrently from the CPU. This approach overlaps GPU recomputation with KV cache transfer to minimize idle GPU time and maximize inference performance. KVPR is fully automated by integrating a profiler module that utilizes input characteristics and system hardware information, a scheduler module to optimize the distribution of computation and communication workloads, and a runtime module to efficiently execute the derived execution plan. Experimental results show that KVPR achieves up to 35.8% lower latency and 46.2% higher throughput during decoding compared to state-of-the-art approaches. The code is available at https://github.com/chaoyij/KVPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04078v1">LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04070v1">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study, hence, focuses on producing precise, in-situ, step-by-step navigation instructions that are practically usable by VI users. Concretely, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to generate rewards guiding the Vision-Language Model (VLM) post-training. This enhances instruction usability while reducing costly real-world data needs. To facilitate training and testing, we introduce NIG4VI, a 27k-sample open-sourced benchmark. It provides diverse navigation scenarios with accurate spatial coordinates, supporting detailed, open-ended in-situ instruction generation. Experiments on NIG4VI show the effectiveness of LaF-GRPO by quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU +14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o's 0.323) and yields more intuitive, safer instructions. Code and benchmark are available at \href{https://github.com/YiyiyiZhao/NIG4VI}{https://github.com/YiyiyiZhao/NIG4VI}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17169v4">REAL: Response Embedding-based Alignment for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) to human preferences is a crucial step in building helpful and safe AI tools, which usually involve training on supervised datasets. Popular algorithms such as Direct Preference Optimization (DPO) rely on pairs of AI-generated responses ranked according to human annotation. The response pair annotation process might bring human bias. Building a correct preference dataset is the costly part of the alignment pipeline. To improve annotation efficiency and quality in the LLMs alignment, we propose REAL: Response Embedding-based Alignment for LLMs, a strategy for constructing a high-quality training dataset that focuses on acquiring the less ambiguous preference pairs for labeling out of a set of response candidates. Our selection process is based on the similarity of embedding responses independently of prompts, which guarantees the selection process in an off-policy setting, avoiding adaptively measuring the similarity during the training. Experimental results on real-world dataset SHP2 and synthetic HH-RLHF benchmarks indicate that choosing dissimilar response pairs enhances the direct alignment of LLMs while reducing inherited labeling errors. The model aligned with dissimilar response pairs obtained a better margin and win rate on the dialogue task. Our findings suggest that focusing on distinct pairs can reduce the label error and improve LLM alignment efficiency, saving up to $65\%$ of annotators' work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04063v1">Crowd-SFT: Crowdsourcing for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) to align model responses with human preferences. While RLHF employs a reinforcement learning approach with a separate reward model, SFT uses human-curated datasets for supervised learning. Both approaches traditionally depend on small, vetted groups of annotators, making them costly, prone to bias, and limited in scalability. We propose an open, crowd-sourced fine-tuning framework that addresses these limitations by enabling broader feedback collection for SFT without extensive annotator training. Our framework promotes incentive fairness via a point-based reward system correlated with Shapley values and guides model convergence through iterative model updates. Our multi-model selection framework demonstrates up to a 55% reduction in target distance over single-model selection, enabling subsequent experiments that validate our point-based reward mechanism's close alignment with Shapley values (a well-established method for attributing individual contributions) thereby supporting fair and scalable participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04051v1">High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04050v1">Explainability-Based Token Replacement on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v3">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02873v1">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23908v2">Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 9 pages, 2 figures, accepted at ACL 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Discovering and evaluating long-form talk content such as videos and podcasts poses a significant challenge for users, as it requires a considerable time investment. Previews offer a practical solution by providing concise snippets that showcase key moments of the content, enabling users to make more informed and confident choices. We propose an LLM-based approach for generating podcast episode previews and deploy the solution at scale, serving hundreds of thousands of podcast previews in a real-world application. Comprehensive offline evaluations and online A/B testing demonstrate that LLM-generated previews consistently outperform a strong baseline built on top of various ML expert models, showcasing a significant reduction in the need for meticulous feature engineering. The offline results indicate notable enhancements in understandability, contextual clarity, and interest level, and the online A/B test shows a 4.6% increase in user engagement with preview content, along with a 5x boost in processing efficiency, offering a more streamlined and performant solution compared to the strong baseline of feature-engineered expert models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02847v1">CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by USENIX ATC 2025
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge devices is crucial for delivering fast responses and ensuring data privacy. However, the limited storage, weight, and power of edge devices make it difficult to deploy LLM-powered applications. These devices must balance latency requirements with energy consumption and model accuracy. In this paper, we first quantify the challenges of deploying LLMs on off-the-shelf edge devices and then we present CLONE, an in-depth algorithm-hardware co-design at both the model- and system-level that intelligently integrates real-time, energy optimization while maintaining robust generality. In order to maximize the synergistic benefits of these algorithms in always-on and intermediate edge computing settings, we specialize in a 28nm scalable hardware accelerator system. We implement and extensively evaluate CLONE on two off-the-shelf edge platforms. Experiments show that CLONE effectively accelerates the inference process up to 11.92x, and saves energy up to 7.36x, while maintaining high-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02818v1">ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive results in natural language processing tasks but require a significant amount of computational and memory resources. Structured matrix representations are a promising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM output is invariant under certain orthogonal transformations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structured classes. The proposed approach is applicable to various types of structured matrices that support efficient projection operations. Code is available at https://github.com/GrishKate/ProcrustesGPT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19146v5">Puzzle: Distillation-Based NAS for Inference-Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer remarkable capabilities, yet their high inference costs restrict wider adoption. While increasing parameter counts improves accuracy, it also broadens the gap between state-of-the-art capabilities and practical deployability. We present Puzzle, a hardware-aware framework that accelerates the inference of LLMs while preserving their capabilities. Using neural architecture search (NAS) at a large-scale, Puzzle optimizes models with tens of billions of parameters. Our approach utilizes blockwise local knowledge distillation (BLD) for parallel architecture exploration and employs mixed-integer programming for precise constraint optimization. We showcase our framework's impact via Llama-3.1-Nemotron-51B-Instruct (Nemotron-51B) and Llama-3.3-Nemotron-49B, two publicly available models derived from Llama-70B-Instruct. Both models achieve a 2.17x inference throughput speedup, fitting on a single NVIDIA H100 GPU while retaining 98.4% of the original model's benchmark accuracies. These are the most accurate models supporting single H100 GPU inference with large batch sizes, despite training on 45B tokens at most, far fewer than the 15T used to train Llama-70B. Lastly, we show that lightweight alignment on these derived models allows them to surpass the parent model in specific capabilities. Our work establishes that powerful LLM models can be optimized for efficient deployment with only negligible loss in quality, underscoring that inference performance, not parameter count alone, should guide model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10927v3">OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 This paper has been accepted to ACM ETRA 2025 and published on PACMHCI
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02758v1">Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications
    </div>
    <details class="paper-abstract">
      Vocabulary use is a fundamental aspect of second language (L2) proficiency. To date, its assessment by automated systems has typically examined the context-independent, or part-of-speech (PoS) related use of words. This paper introduces a novel approach to enable fine-grained vocabulary evaluation exploiting the precise use of words within a sentence. The scheme combines large language models (LLMs) with the English Vocabulary Profile (EVP). The EVP is a standard lexical resource that enables in-context vocabulary use to be linked with proficiency level. We evaluate the ability of LLMs to assign proficiency levels to individual words as they appear in L2 learner writing, addressing key challenges such as polysemy, contextual variation, and multi-word expressions. We compare LLMs to a PoS-based baseline. LLMs appear to exploit additional semantic information that yields improved performance. We also explore correlations between word-level proficiency and essay-level proficiency. Finally, the approach is applied to examine the consistency of the EVP proficiency levels. Results show that LLMs are well-suited for the task of vocabulary assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23754v2">DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01586v2">SubTrack++ : Gradient Subspace Tracking for Scalable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) is highly resource-intensive due to their massive number of parameters and the overhead of optimizer states. While recent work has aimed to reduce memory consumption, such efforts often entail trade-offs among memory efficiency, training time, and model performance. Yet, true democratization of LLMs requires simultaneous progress across all three dimensions. To this end, we propose SubTrack++ that leverages Grassmannian gradient subspace tracking combined with projection-aware optimizers, enabling Adam's internal statistics to adapt to changes in the optimization subspace. Additionally, employing recovery scaling, a technique that restores information lost through low-rank projections, further enhances model performance. Our method demonstrates SOTA convergence by exploiting Grassmannian geometry and achieves lowest evaluation loss, outperforming the current SOTA while reducing pretraining wall time by 43% and maintaining the memory footprint on a 1B-parameter Llama model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02742v1">Prompt-Unseen-Emotion: Zero-shot Expressive Speech Synthesis with Prompt-LLM Contextual Knowledge for Mixed Emotions</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Existing expressive text-to-speech (TTS) systems primarily model a limited set of categorical emotions, whereas human conversations extend far beyond these predefined emotions, making it essential to explore more diverse emotional speech generation for more natural interactions. To bridge this gap, this paper proposes a novel prompt-unseen-emotion (PUE) approach to generate unseen emotional speech via emotion-guided prompt learning. PUE is trained utilizing an LLM-TTS architecture to ensure emotional consistency between categorical emotion-relevant prompts and emotional speech, allowing the model to quantitatively capture different emotion weightings per utterance. During inference, mixed emotional speech can be generated by flexibly adjusting emotion proportions and leveraging LLM contextual knowledge, enabling the model to quantify different emotional styles. Our proposed PUE successfully facilitates expressive speech synthesis of unseen emotions in a zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02718v1">Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across diverse natural language processing tasks, yet their deployment in real-world applications is hindered by fixed knowledge cutoffs and difficulties in generating controllable, accurate outputs in a single inference. Multi-agent systems (MAS) built from specialized LLM agents offer a promising solution, enabling dynamic collaboration and iterative reasoning. However, optimizing these systems remains a challenge, as conventional methods such as prompt engineering and supervised fine-tuning entail high engineering overhead and limited adaptability. Reinforcement learning (RL), particularly multi-agent reinforcement learning (MARL), provides a scalable framework by refining agent policies based on system-level feedback. Nevertheless, existing MARL algorithms, such as Multi-Agent Proximal Policy Optimization (MAPPO), rely on Critic networks, which can cause training instability and increase computational burden. To address these limitations and target the prototypical Multi-Agent Search System (MASS), we propose Multi-Agent Heterogeneous Group Policy Optimization (MHGPO), a novel Critic-free algorithm that guides policy updates by estimating relative reward advantages across heterogeneous groups of rollouts. MHGPO eliminates the need for Critic networks, enhancing stability and reducing computational overhead. Additionally, we introduce three group rollout sampling strategies that trade off between efficiency and effectiveness. Experiments on a multi-agent LLM-based search system demonstrate that MHGPO consistently outperforms MAPPO in both task performance and computational efficiency, without requiring warm-up, underscoring its potential for stable and scalable optimization of complex LLM-based MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17878v2">Towards Enhanced Immersion and Agency for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL'2025
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00519v2">CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to Association for Computational Linguistics Findings (ACL) 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at https://github.com/peachch/CausalAbstain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.10424v3">LLM-as-an-Interviewer: Beyond Static Testing Through Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We introduce LLM-as-an-Interviewer, a novel paradigm for evaluating large language models (LLMs). This approach leverages multi-turn interactions where the LLM interviewer actively provides feedback on responses and poses follow-up questions to the evaluated LLM. At the start of the interview, the LLM interviewer dynamically modifies datasets to generate initial questions, mitigating data contamination. We apply the LLM-as-an-Interviewer framework to evaluate six models on the MATH and DepthQA tasks. Our results show that the framework effectively provides insights into LLM performance, including the quality of initial responses, adaptability to feedback, and ability to address follow-up queries like clarification or additional knowledge requests. The framework also addresses key limitations of conventional methods like LLM-as-a-Judge, including verbosity bias and inconsistency across runs. Finally, we propose the Interview Report, which aggregates insights from the interview process, providing examples and a comprehensive analysis of the LLM's strengths and weaknesses. This report offers a detailed snapshot of the model's real-world applicability. The code for our framework is publicly available at https://github.com/interview-eval/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.11674v2">LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Time series forecasting using LLMs
    </div>
    <details class="paper-abstract">
      Time series forecasting remains a challenging task, particularly in the context of complex multiscale temporal patterns. This study presents LLM-Mixer, a framework that improves forecasting accuracy through the combination of multiscale time-series decomposition with pre-trained LLMs (Large Language Models). LLM-Mixer captures both short-term fluctuations and long-term trends by decomposing the data into multiple temporal resolutions and processing them with a frozen LLM, guided by a textual prompt specifically designed for time-series data. Extensive experiments conducted on multivariate and univariate datasets demonstrate that LLM-Mixer achieves competitive performance, outperforming recent state-of-the-art models across various forecasting horizons. This work highlights the potential of combining multiscale analysis and LLMs for effective and scalable time-series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.18458v3">A Survey of LLM $\times$ DATA</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Please refer to the paper list at: https://github.com/weAIDB/awesome-data-llm
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.18547v5">Token-Budget-Aware LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning and enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework that dynamically adjusts the number of reasoning tokens based on the reasoning complexity of each problem. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15146v2">lmgame-Bench: How Good are LLMs at Playing Games?</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Playing video games requires perception, memory, and planning, exactly the faculties modern large language model (LLM) agents are expected to master. We study the major challenges in using popular video games to evaluate modern LLMs and find that directly dropping LLMs into games cannot make an effective evaluation, for three reasons -- brittle vision perception, prompt sensitivity, and potential data contamination. We introduce lmgame-Bench to turn games into reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and narrative games delivered through a unified Gym-style API and paired with lightweight perception and memory scaffolds, and is designed to stabilize prompt variance and remove contamination. Across 13 leading models, we show lmgame-Bench is challenging while still separating models well. Correlation analysis shows that every game probes a unique blend of capabilities often tested in isolation elsewhere. More interestingly, performing reinforcement learning on a single game from lmgame-Bench transfers both to unseen games and to external planning tasks. Our evaluation code is available at https://github.com/lmgame-org/GamingAgent/lmgame-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02696v1">Shaking to Reveal: Perturbation-Based Detection of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Hallucination remains a key obstacle to the reliable deployment of large language models (LLMs) in real-world question answering tasks. A widely adopted strategy to detect hallucination, known as self-assessment, relies on the model's own output confidence to estimate the factual accuracy of its answers. However, this strategy assumes that the model's output distribution closely reflects the true data distribution, which may not always hold in practice. As bias accumulates through the model's layers, the final output can diverge from the underlying reasoning process, making output-level confidence an unreliable signal for hallucination detection. In this work, we propose Sample-Specific Prompting (SSP), a new framework that improves self-assessment by analyzing perturbation sensitivity at intermediate representations. These representations, being less influenced by model bias, offer a more faithful view of the model's latent reasoning process. Specifically, SSP dynamically generates noise prompts for each input and employs a lightweight encoder to amplify the changes in representations caused by the perturbation. A contrastive distance metric is then used to quantify these differences and separate truthful from hallucinated responses. By leveraging the dynamic behavior of intermediate representations under perturbation, SSP enables more reliable self-assessment. Extensive experiments demonstrate that SSP significantly outperforms prior methods across a range of hallucination detection benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02678v1">TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15319v2">GPTVQ: The Blessing of Dimensionality for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      In this work we show that the size versus accuracy trade-off of neural network quantization can be significantly improved by increasing the quantization dimensionality. We propose the GPTVQ method, a new fast method for post-training vector quantization (VQ) that scales well to Large Language Models (LLMs). Our method interleaves quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE. Quantization codebooks are initialized using an efficient data-aware version of the EM algorithm. The codebooks are then updated, and further compressed by using integer quantization and SVD-based compression. GPTVQ establishes a new state-of-the art in the size vs accuracy trade-offs on a wide range of LLMs such as Llama-v2 and Mistral. Furthermore, our method is efficient: on a single H100 it takes between 3 and 11 hours to process a Llamav2-70B model, depending on quantization setting. Lastly, with on-device timings for VQ decompression on a mobile CPU we show that VQ leads to improved latency compared to using a 4-bit integer format.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02672v1">EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 47 pages, 24 figures
    </div>
    <details class="paper-abstract">
      We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02659v1">Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) are increasingly used in diverse applications, where they are assigned a specific persona - such as a happy high school teacher - to guide their responses. While prior research has examined how well LLMs adhere to predefined personas in writing style, a comprehensive analysis of consistency across different personas and task types is lacking. In this paper, we introduce a new standardized framework to analyze consistency in persona-assigned LLMs. We define consistency as the extent to which a model maintains coherent responses when assigned the same persona across different tasks and runs. Our framework evaluates personas across four different categories (happiness, occupation, personality, and political stance) spanning multiple task dimensions (survey writing, essay generation, social media post generation, single turn, and multi-turn conversations). Our findings reveal that consistency is influenced by multiple factors, including the assigned persona, stereotypes, and model design choices. Consistency also varies across tasks, increasing with more structured tasks and additional context. All code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21271v4">EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA, a novel fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior training-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{6.74\%}$ on GSM8K) for LLaMA3-8B compressed to 3-bit. We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07965v5">SHARP: Unlocking Interactive Hallucination via Stance Transfer in Role-Playing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 28 pages, unfortunately accepted to findings with Meta 4, acknowledge and apologize to the reviewers and area chair who support our work in the discussion period
    </div>
    <details class="paper-abstract">
      The advanced role-playing capabilities of Large Language Models (LLMs) have enabled rich interactive scenarios, yet existing research in social interactions neglects hallucination while struggling with poor generalizability and implicit character fidelity judgments. To bridge this gap, motivated by human behaviour, we introduce a generalizable and explicit paradigm for uncovering interactive patterns of LLMs across diverse worldviews. Specifically, we first define interactive hallucination through stance transfer, then construct SHARP, a benchmark built by extracting relations from commonsense knowledge graphs and utilizing LLMs' inherent hallucination properties to simulate multi-role interactions. Extensive experiments confirm our paradigm's effectiveness and stability, examine the factors that influence these metrics, and challenge conventional hallucination mitigation solutions. More broadly, our work reveals a fundamental limitation in popular post-training methods for role-playing LLMs: the tendency to obscure knowledge beneath style, resulting in monotonous yet human-like behaviors - interactive hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24539v2">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01926v2">Unnatural Languages Are Not Bugs but Features for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been observed to process non-human-readable text sequences, such as jailbreak prompts, often viewed as a bug for aligned LLMs. In this work, we present a systematic investigation challenging this perception, demonstrating that unnatural languages - strings that appear incomprehensible to humans but maintain semantic meanings for LLMs - contain latent features usable by models. Notably, unnatural languages possess latent features that can be generalized across different models and tasks during inference. Furthermore, models fine-tuned on unnatural versions of instruction datasets perform on-par with those trained on natural language, achieving 49.71 win rates in Length-controlled AlpacaEval 2.0 in average across various base models. In addition, through comprehensive analysis, we demonstrate that LLMs process unnatural languages by filtering noise and inferring contextual meaning from filtered words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02592v1">Beyond the Surface: Measuring Self-Preference in LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent studies show that large language models (LLMs) exhibit self-preference bias when serving as judges, meaning they tend to favor their own responses over those generated by other models. Existing methods typically measure this bias by calculating the difference between the scores a judge model assigns to its own responses and those it assigns to responses from other models. However, this approach conflates self-preference bias with response quality, as higher-quality responses from the judge model may also lead to positive score differences, even in the absence of bias. To address this issue, we introduce gold judgments as proxies for the actual quality of responses and propose the DBG score, which measures self-preference bias as the difference between the scores assigned by the judge model to its own responses and the corresponding gold judgments. Since gold judgments reflect true response quality, the DBG score mitigates the confounding effect of response quality on bias measurement. Using the DBG score, we conduct comprehensive experiments to assess self-preference bias across LLMs of varying versions, sizes, and reasoning abilities. Additionally, we investigate two factors that influence and help alleviate self-preference bias: response text style and the post-training data of judge models. Finally, we explore potential underlying mechanisms of self-preference bias from an attention-based perspective. Our code and data are available at https://github.com/zhiyuanc2001/self-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02591v1">On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to ACL 2025 Main (Camera-Ready Version)
    </div>
    <details class="paper-abstract">
      Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: (RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02589v1">Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17626v3">Tracking the Feature Dynamics in LLM Training: A Mechanistic Study</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we (1) introduce SAE-Track, a novel method for efficiently obtaining a continual series of SAEs, providing the foundation for a mechanistic study that covers (2) the semantic evolution of features, (3) the underlying processes of feature formation, and (4) the directional drift of feature vectors. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution. For reproducibility, our code is available at https://github.com/Superposition09m/SAE-Track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02573v1">IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Although region-specific large language models (LLMs) are increasingly developed, their safety remains underexplored, particularly in culturally diverse settings like Indonesia, where sensitivity to local norms is essential and highly valued by the community. In this work, we present IndoSafety, the first high-quality, human-verified safety evaluation dataset tailored for the Indonesian context, covering five language varieties: formal and colloquial Indonesian, along with three major local languages: Javanese, Sundanese, and Minangkabau. IndoSafety is constructed by extending prior safety frameworks to develop a taxonomy that captures Indonesia's sociocultural context. We find that existing Indonesian-centric LLMs often generate unsafe outputs, particularly in colloquial and local language settings, while fine-tuning on IndoSafety significantly improves safety while preserving task performance. Our work highlights the critical need for culturally grounded safety evaluation and provides a concrete step toward responsible LLM deployment in multilingual settings. Warning: This paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01538v2">LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at https://windylab.github.io/LAMARL/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04291v2">MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Published in AAAI 2025
    </div>
    <details class="paper-abstract">
      We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06705v3">LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02553v1">Response-Level Rewards Are All You Need for Online Reinforcement Learning in LLMs: A Mathematical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We study a common challenge in reinforcement learning for large language models (LLMs): the Zero-Reward Assumption, where non-terminal actions (i.e., intermediate token generations) receive zero task-specific immediate reward, while only the final token receives a reward for the entire response. This assumption arises frequently in practice, as precise token-level rewards are often difficult or infeasible to obtain in LLM applications. In this work, we provide a unifying theoretical perspective. We introduce the Trajectory Policy Gradient Theorem, which shows that the policy gradient based on true, unknown token-level rewards can be unbiasedly estimated using only a response-level reward model, regardless of whether the Zero-Reward Assumption holds or not, for algorithms in the REINFORCE and Actor-Critic families. This result reveals that widely used methods such as PPO, GRPO, ReMax, and RLOO inherently possess the capacity to model token-level reward signals, offering a theoretical justification for response-level reward approaches. Our findings pave the way for more practical, efficient LLM fine-tuning, allowing developers to treat training algorithms as black boxes and focus on improving the response-level reward model with auxiliary sub-models. We also offer a detailed analysis of popular RL and non-RL methods, comparing their theoretical foundations and practical advantages across common LLM tasks. Finally, we propose a new algorithm: Token-Reinforced Policy Optimization (TRePO), a theoretically grounded method that is simpler than PPO, matches GRPO in memory efficiency, and holds promise for broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23809v2">LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02546v1">Attention Knows Whom to Trust: Attention-based Trust Management for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have demonstrated strong capabilities in solving complex tasks but remain vulnerable when agents receive unreliable messages. This vulnerability stems from a fundamental gap: LLM agents treat all incoming messages equally without evaluating their trustworthiness. While some existing studies approach the trustworthiness, they focus on a single type of harmfulness rather than analyze it in a holistic approach from multiple trustworthiness perspectives. In this work, we propose Attention Trust Score (A-Trust), a lightweight, attention-based method for evaluating message trustworthiness. Inspired by human communication literature[1], through systematically analyzing attention behaviors across six orthogonal trust dimensions, we find that certain attention heads in the LLM specialize in detecting specific types of violations. Leveraging these insights, A-Trust directly infers trustworthiness from internal attention patterns without requiring external prompts or verifiers. Building upon A-Trust, we develop a principled and efficient trust management system (TMS) for LLM-MAS, enabling both message-level and agent-level trust assessment. Experiments across diverse multi-agent settings and tasks demonstrate that applying our TMS significantly enhances robustness against malicious inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18499v2">G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated remarkable progress, their proficiency in graph-related tasks remains notably limited, hindering the development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face challenges such as the scarcity of large-scale, universally represented graph data. We introduce G1, a simple yet effective approach demonstrating that Reinforcement Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs' graph reasoning abilities. To enable RL training, we curate Erd\~os, the largest graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of varying difficulty levels, 100k training data and 5k test data, all drived from real-world graphs. With RL on Erd\~os, G1 obtains substantial improvements in graph reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct (24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic benchmarks as well as real-world node classification and link prediction tasks, without compromising general reasoning abilities. Our findings offer an efficient, scalable path for building strong graph reasoners by finetuning LLMs with RL on graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs possess graph understanding abilities that RL can elicit successfully. Our implementation is open-sourced at https://github.com/PKU-ML/G1, with models and datasets hosted on Hugging Face collections https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2 for broader accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02522v1">Think Twice, Act Once: A Co-Evolution Framework of LLM and RL for Large-Scale Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and Reinforcement Learning (RL) have shown significant promise in decision-making tasks. Nevertheless, for large-scale industrial decision problems, both approaches face distinct challenges: LLMs lack real-time long-sequence decision-making capabilities, while RL struggles with sample efficiency in vast action spaces. To bridge this gap, we propose Agents Co-Evolution (ACE), a synergistic framework between LLMs and RL agents for large-scale decision-making scenarios. ACE introduces a dual-role trajectory refinement mechanism where LLMs act as both Policy Actor and Value Critic during RL's training: the Actor refines suboptimal actions via multi-step reasoning and environment validation, while the Critic performs temporal credit assignment through trajectory-level reward shaping. Concurrently, RL agent enhances LLMs' task-specific decision-making with high-quality fine-tuning datasets generated via prioritized experience replay. Through extensive experiments across multiple power grid operation challenges with action spaces exceeding 60K discrete actions, ACE demonstrates superior performance over existing RL methods and LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02519v1">Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted at ACL Main 2025
    </div>
    <details class="paper-abstract">
      LLMssuch as GPT-4 have shown a remarkable ability to solve complex questions by generating step-by-step rationales. Prior works have utilized this capability to improve smaller and cheaper LMs (say, with 7B parameters). However, various practical constraints, such as copyright and legal issues, owing to lack of transparency in the pre-training data of large (often closed) models, prevent their use in commercial settings. Little focus has been given to improving the innate reasoning ability of smaller models without distilling information from larger LLMs. To address this, we propose COLLATE, a trainable framework that tunes a (small) LLM to generate those outputs from a pool of diverse rationales that selectively improves the downstream task. COLLATE enforces multiple instances of the same LLM to exhibit distinct behavior and employs them to generate rationales to obtain diverse outputs. The LLM is then tuned via preference optimization to choose the candidate rationale which maximizes the likelihood of ground-truth answer. COLLATE outperforms several trainable and prompting baselines on 5 datasets across 3 domains: maths problem solving, natural language inference, and commonsense reasoning. We show the eff icacy of COLLATE on LLMs from different model families across varying parameter scales (1B to 8B) and demonstrate the benefit of multiple rationale providers guided by the end task through ablations. Code is released here (https://github.com/Sohanpatnaik106/collate).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02514v1">To Embody or Not: The Effect Of Embodiment On User Perception Of LLM-based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Embodiment in conversational agents (CAs) refers to the physical or visual representation of these agents, which can significantly influence user perception and interaction. Limited work has been done examining the effect of embodiment on the perception of CAs utilizing modern large language models (LLMs) in non-hierarchical cooperative tasks, a common use case of CAs as more powerful models become widely available for general use. To bridge this research gap, we conducted a mixed-methods within-subjects study on how users perceive LLM-based CAs in cooperative tasks when embodied and non-embodied. The results show that the non-embodied agent received significantly better quantitative appraisals for competence than the embodied agent, and in qualitative feedback, many participants believed that the embodied CA was more sycophantic than the non-embodied CA. Building on prior work on users' perceptions of LLM sycophancy and anthropomorphic features, we theorize that the typically-positive impact of embodiment on perception of CA credibility can become detrimental in the presence of sycophancy. The implication of such a phenomenon is that, contrary to intuition and existing literature, embodiment is not a straightforward way to improve a CA's perceived credibility if there exists a tendency to sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00486v2">It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: https://huggingface.co/spaces/shifeng3711/gg_prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02490v1">Simplifying Root Cause Analysis in Kubernetes with StateGraph and LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 12 pages, 13 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Kubernetes, a notably complex and distributed system, utilizes an array of controllers to uphold cluster management logic through state reconciliation. Nevertheless, maintaining state consistency presents significant challenges due to unexpected failures, network disruptions, and asynchronous issues, especially within dynamic cloud environments. These challenges result in operational disruptions and economic losses, underscoring the necessity for robust root cause analysis (RCA) to enhance Kubernetes reliability. The development of large language models (LLMs) presents a promising direction for RCA. However, existing methodologies encounter several obstacles, including the diverse and evolving nature of Kubernetes incidents, the intricate context of incidents, and the polymorphic nature of these incidents. In this paper, we introduce SynergyRCA, an innovative tool that leverages LLMs with retrieval augmentation from graph databases and enhancement with expert prompts. SynergyRCA constructs a StateGraph to capture spatial and temporal relationships and utilizes a MetaGraph to outline entity connections. Upon the occurrence of an incident, an LLM predicts the most pertinent resource, and SynergyRCA queries the MetaGraph and StateGraph to deliver context-specific insights for RCA. We evaluate SynergyRCA using datasets from two production Kubernetes clusters, highlighting its capacity to identify numerous root causes, including novel ones, with high efficiency and precision. SynergyRCA demonstrates the ability to identify root causes in an average time of about two minutes and achieves an impressive precision of approximately 0.90.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19390v2">Checkpoint Merging via Bayesian Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      The rapid proliferation of large language models (LLMs) such as GPT-4 and Gemini underscores the intense demand for resources during their training processes, posing significant challenges due to substantial computational and environmental costs. To alleviate this issue, we propose checkpoint merging in pretraining LLM. This method utilizes LLM checkpoints with shared training trajectories, and is rooted in an extensive search space exploration for the best merging weight via Bayesian optimization. Through various experiments, we demonstrate that: (1) Our proposed methodology exhibits the capacity to augment pretraining, presenting an opportunity akin to obtaining substantial benefits at minimal cost; (2) Our proposed methodology, despite requiring a given held-out dataset, still demonstrates robust generalization capabilities across diverse domains, a pivotal aspect in pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v4">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v7">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Tauric Research @ https://github.com/TauricResearch; Oral @ Multi-Agent AI in the Real World
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, the multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/TauricResearch/TradingAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02470v1">A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Misdiagnosis causes significant harm to healthcare systems worldwide, leading to increased costs and patient risks. MedRAG is a smart multimodal healthcare copilot equipped with powerful large language model (LLM) reasoning, designed to enhance medical decision-making. It supports multiple input modalities, including non-intrusive voice monitoring, general medical queries, and electronic health records. MedRAG provides recommendations on diagnosis, treatment, medication, and follow-up questioning. Leveraging retrieval-augmented generation enhanced by knowledge graph-elicited reasoning, MedRAG retrieves and integrates critical diagnostic insights, reducing the risk of misdiagnosis. It has been evaluated on both public and private datasets, outperforming existing models and offering more specific and accurate healthcare assistance. A demonstration video of MedRAG is available at: https://www.youtube.com/watch?v=PNIBDMYRfDM. The source code is available at: https://github.com/SNOWTEAM2023/MedRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13508v2">Time-R1: Towards Comprehensive Temporal Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02457v1">SOVA-Bench: Benchmarking the Speech Conversation Ability for LLM-based Voice Assistant</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Thanks to the steady progress of large language models (LLMs), speech encoding algorithms and vocoder structure, recent advancements have enabled generating speech response directly from a user instruction. However, benchmarking the generated speech quality has been a neglected but critical issue, considering the shift from the pursuit of semantic accuracy to vivid and spontaneous speech flow. Previous evaluation focused on the speech-understanding ability, lacking a quantification of acoustic quality. In this paper, we propose Speech cOnversational Voice Assistant Benchmark (SOVA-Bench), providing a comprehension comparison of the general knowledge, speech recognition and understanding, along with both semantic and acoustic generative ability between available speech LLMs. To the best of our knowledge, SOVA-Bench is one of the most systematic evaluation frameworks for speech LLMs, inspiring the direction of voice interaction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02442v1">Should LLM Safety Be More Than Refusing Harmful Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00418v3">Self-Evolved Reward Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 23 pages,6 figures,Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. These methods can be costly and may introduce biases that affect the language model's responses. As language models improve, human input may become less effective in further enhancing their performance. In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs). Resources of this paper can be found at https://aka.ms/ser
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17214v2">CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on Llama Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: https://github.com/ZBox1005/CoT-UQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v3">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00694v2">Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 11 pages, 7th Workshop on Automated Semantic Analysis of Information in Legal Text @ ICAIL 2025, 16 June 2025, Chicago, IL
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Link: https://lizhang-aiandlaw.github.io/An-Automated-Pipeline-for-Evaluating-LLM-Generated-3-ply-Case-Based-Legal-Arguments/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00490v2">LLM-Driven Instance-Specific Heuristic Generation and Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Combinatorial optimization problems are widely encountered in real-world applications. Designing high-quality heuristic algorithms that efficiently approximate optimal solutions within reasonable time is a critical research challenge. In recent years, many works have explored integrating Large Language Models (LLMs) with Evolutionary Algorithms to automate heuristic algorithm design through prompt engineering. However, these approaches generally adopt a problem-specific paradigm, applying a single algorithm across all problem instances, failing to account for the heterogeneity across instances. In this paper, we propose InstSpecHH, a novel framework that introduces the concept of instance-specific heuristic generation. InstSpecHH partitions the overall problem class into sub-classes based on instance features and performs differentiated, automated heuristic design for each problem subclass. By tailoring heuristics to the unique features of different sub-classes, InstSpecHH achieves better performance at the problem class level while avoiding redundant heuristic generation for similar instances, thus reducing computational overhead. This approach effectively balances the trade-off between the cost of automatic heuristic design and the quality of the obtained solutions. To evaluate the performance of InstSpecHH, we conduct experiments on 4,500 subclasses of the Online Bin Packing Problem (OBPP) and 365 subclasses of the Capacitated Vehicle Routing Problem (CVRP). Experimental results show that InstSpecHH demonstrates strong intra-subclass and inter-subclass generalization capabilities. Compared to previous problem-specific methods, InstSpecHH reduces the average optimality gap by more than 5.6\% for OBPP and 0.9\% for CVRP. These results highlight the potential of instance-aware automatic heuristic design to further enhance solution quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02389v1">Univariate to Multivariate: LLMs as Zero-Shot Predictors for Time-Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Time-series prediction or forecasting is critical across many real-world dynamic systems, and recent studies have proposed using Large Language Models (LLMs) for this task due to their strong generalization capabilities and ability to perform well without extensive pre-training. However, their effectiveness in handling complex, noisy, and multivariate time-series data remains underexplored. To address this, we propose LLMPred which enhances LLM-based time-series prediction by converting time-series sequences into text and feeding them to LLMs for zero shot prediction along with two main data pre-processing techniques. First, we apply time-series sequence decomposition to facilitate accurate prediction on complex and noisy univariate sequences. Second, we extend this univariate prediction capability to multivariate data using a lightweight prompt-processing strategy. Extensive experiments with smaller LLMs such as Llama 2 7B, Llama 3.2 3B, GPT-4o-mini, and DeepSeek 7B demonstrate that LLMPred achieves competitive or superior performance compared to state-of-the-art baselines. Additionally, a thorough ablation study highlights the importance of the key components proposed in LLMPred.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00095v2">ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02372v1">AnswerCarefully: A Dataset for Improving the Safety of Japanese LLM Output</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      In this paper we present AnswerCarefully, a dataset for promoting the safety and appropriateness of Japanese LLM outputs. The dataset consists of 1,800 pairs of questions and reference answers, where the questions require special attention in answering. It covers a wide range of risk categories established in prior English-language datasets, but the data samples are original in that they are manually created to reflect the socio-cultural context of LLM usage in Japan. We show that using this dataset for instruction to fine-tune a Japanese LLM led to improved output safety without compromising the utility of general responses. We also report the results of a safety evaluation of 12 Japanese LLMs using this dataset as a benchmark. Finally, we describe the latest update on the dataset which provides English translations and annotations of the questions, aimed at facilitating the derivation of similar datasets in different languages and regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v4">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02368v1">NextQuill: Causal Preference Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) for individual users has become increasingly important as they are progressively integrated into real-world applications to support users' daily lives. However, existing personalization approaches often fail to distinguish which components of model predictions and training data truly reflect user preferences, leading to superficial personalization alignment. In this paper, we introduce NextQuill, a novel LLM personalization alignment framework grounded in causal preference modeling. We approach personalization from a causal perspective, treating both model predictions and ground-truth data generation as outcomes influenced by user preferences, along with other factors. We define the true preference effect as the causal impact of user history (which reflects preferences) on each token prediction or data generation instance, estimated through causal intervention techniques. Building on this insight, NextQuill introduces two complementary alignment strategies: (1) aligning model-internal causal preference effects on predictions with those reflected in ground-truth data, rather than indiscriminately fitting predictions, and (2) focusing on fitting preference-bearing tokens identified via ground-truth data preference effects, rather than treating all tokens uniformly. By integrating these strategies, NextQuill shifts the alignment process toward learning from causal preference effects, facilitating more effective and personalized adaptation. Experiments across multiple personalization benchmarks demonstrate that NextQuill significantly improves personalization quality, offering a principled, causal foundation for LLM personalization. Our codes are available on https://github.com/juntaoyou/NextQuill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02357v1">Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint. This work has been submitted to the Technical AI Governance Workshop at ICML 2025 for review
    </div>
    <details class="paper-abstract">
      Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. Failure to prioritize such principles indicates a potential basic control failure. This paper introduces a lightweight, interpretable benchmark methodology using a simple grid world to evaluate an LLM agent's ability to uphold a predefined, high-level safety principle (e.g., "never enter hazardous zones") when faced with conflicting lower-level task instructions. We probe whether the agent reliably prioritizes the inviolable directive, testing a foundational controllability aspect of LLMs. This pilot study demonstrates the methodology's feasibility, offers preliminary insights into agent behavior under principle conflict, and discusses how such benchmarks can contribute empirical evidence for assessing controllability. We argue that evaluating adherence to hierarchical principles is a crucial early step in understanding our capacity to build governable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02351v1">DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 To appear in the First REALM (Research on Agent Language Models) workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02338v1">One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 ACL 2025 Industry
    </div>
    <details class="paper-abstract">
      With the release of R1, a publicly available large reasoning model (LRM), researchers commonly train new LRMs by training language models on R1's long chain-of-thought (CoT) inferences. While prior works show that LRMs' capabilities can be reproduced through direct distillation, the continued reliance on the existing models (e.g., R1) remains a critical limitation in advancing the field. As a first step toward independent LRM development, this paper explores the possibility of constructing a long CoT dataset with LLMs that are not trained for inference-time scaling. To this end, we present the Long CoT Collection, a dataset of 100K CoT rationales annotated using existing short CoT LLMs. We develop a pipeline that induces o1's novel reasoning strategies into short CoT LLMs, enabling them to think longer and introducing controllability over the thought budget to better manage the overthinking problem. Our extensive analyses validate that our dataset achieves quality comparable to--or slightly below--R1. Furthermore, our experiments demonstrate that training on our dataset not only strengthens general reasoning skills, but also provides a strong foundation for reinforcement learning--models initialized on our data achieve 2-3x larger gains with RLVR.
    </details>
</div>
