# llm - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01612v1">Ontology Population using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-03
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs) are increasingly utilized for data integration, representation, and visualization. While KG population is critical, it is often costly, especially when data must be extracted from unstructured text in natural language, which presents challenges, such as ambiguity and complex interpretations. Large Language Models (LLMs) offer promising capabilities for such tasks, excelling in natural language understanding and content generation. However, their tendency to ``hallucinate'' can produce inaccurate outputs. Despite these limitations, LLMs offer rapid and scalable processing of natural language data, and with prompt engineering and fine-tuning, they can approximate human-level performance in extracting and structuring data for KGs. This study investigates LLM effectiveness for the KG population, focusing on the Enslaved.org Hub Ontology. In this paper, we report that compared to the ground truth, LLM's can extract ~90% of triples, when provided a modular ontology as guidance in the prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00799v5">Are you still on track!? Catching LLM Task Drift with Activations</a></div>
    <div class="paper-meta">
      📅 2024-11-03
    </div>
    <details class="paper-abstract">
      Large Language Models are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how user's tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and inspection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01562v1">Are LLMs good pragmatic speakers?</a></div>
    <div class="paper-meta">
      📅 2024-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on data assumed to include natural language pragmatics, but do they actually behave like pragmatic speakers? We attempt to answer this question using the Rational Speech Act (RSA) framework, which models pragmatic reasoning in human communication. Using the paradigm of a reference game constructed from the TUNA corpus, we score candidate referential utterances in both a state-of-the-art LLM (Llama3-8B-Instruct) and in the RSA model, comparing and contrasting these scores. Given that RSA requires defining alternative utterances and a truth-conditional meaning function, we explore such comparison for different choices of each of these requirements. We find that while scores from the LLM have some positive correlation with those from RSA, there isn't sufficient evidence to claim that it behaves like a pragmatic speaker. This initial study paves way for further targeted efforts exploring different models and settings, including human-subject evaluation, to see if LLMs truly can, or be made to, behave like pragmatic speakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v1">LumosCore: Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2024-11-03
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Model(LLM) technologies has led to a rapidly growing demand for compute resources in models. In response, the enterprises are building large-scale multi-tenant GPU clusters with 10k or even ore GPUs. In contrast to the rapidly growing cluster size, the bandwidth of clusters has also been increasing to meet communication demands, with 800 Gbps optical modules already in practical use and 1.6 Tbps modules on the horizon. However, designing clusters that simultaneously meet the requirements of large scale and high bandwidth is challenging due to the limited capacity of electrical switch chips. Unlike electrical switch chips, the single-port bandwidth of MEMS-OCS is solely determined by the optical module, making it straightforward to achieve both bandwidth and scability requirement. In this paper, we propose an opto-electronic hybrid architecture called \textbf{LumosCore}. We address the issues of L2 protocols incompatibility potential network contention and algorithm time complexity through physical topology and logical topology design. Additionally, we design a polynomial-time complexity link reconfiguration algorithm to reconfigure MEMS-OCS with minimal time overhead. We validate the feasibility of the proposed scheme in a cluster consisting of 128 NPUs, and through simulation based on real traces, we demonstrate the superiority of \textbf{LumosCore} over traditional architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09395v6">Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions</a></div>
    <div class="paper-meta">
      📅 2024-11-03
      | 💬 With o1 and GPT-4o results. Reformatted the data and presented more analysis
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have showcased striking results on existing logical reasoning benchmarks, with some models even surpassing human performance. However, the true depth of their competencies and robustness in reasoning tasks remains an open question. To this end, in this paper, we focus on two popular reasoning tasks: arithmetic reasoning and code generation. Particularly, we introduce (i) a general ontology of perturbations for math and coding questions, (ii) a semi-automatic method to apply these perturbations, and (iii) two datasets, GSMORE and HUMANEVAL-CORE, respectively, of perturbed math and coding problems to probe LLM capabilities in numeric reasoning and coding tasks. Through comprehensive evaluations of both closed-source and open-source LLMs, we show a significant performance drop across all the models against the perturbed questions, suggesting that the current LLMs lack robust problem solving skills and structured reasoning abilities in many areas, as defined by our ontology. We open-source the datasets and source codes at: https://github.com/declare-lab/LLM-ReasoningTest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15481v2">Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize code-switching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand code-switching texts. Additionally, we validate the extensibility of the CSRT by generating code-switching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13042v2">StoryVerse: Towards Co-authoring Dynamic Plot with LLM-based Character Simulation via Narrative Planning</a></div>
    <div class="paper-meta">
      📅 2024-11-02
      | 💬 Proceedings of the 19th international conference on the foundations of digital games 2024
    </div>
    <details class="paper-abstract">
      Automated plot generation for games enhances the player's experience by providing rich and immersive narrative experience that adapts to the player's actions. Traditional approaches adopt a symbolic narrative planning method which limits the scale and complexity of the generated plot by requiring extensive knowledge engineering work. Recent advancements use Large Language Models (LLMs) to drive the behavior of virtual characters, allowing plots to emerge from interactions between characters and their environments. However, the emergent nature of such decentralized plot generation makes it difficult for authors to direct plot progression. We propose a novel plot creation workflow that mediates between a writer's authorial intent and the emergent behaviors from LLM-driven character simulation, through a novel authorial structure called "abstract acts". The writers define high-level plot outlines that are later transformed into concrete character action sequences via an LLM-based narrative planning process, based on the game world state. The process creates "living stories" that dynamically adapt to various game world states, resulting in narratives co-created by the author, character simulation, and player. We present StoryVerse as a proof-of-concept system to demonstrate this plot creation workflow. We showcase the versatility of our approach with examples in different stories and game environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01142v1">NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      Online LLM inference powers many exciting applications such as intelligent chatbots and autonomous agents. Modern LLM inference engines widely rely on request batching to improve inference throughput, aiming to make it cost-efficient when running on expensive GPU accelerators. However, the limited GPU memory has largely limited the batch size achieved in practice, leaving significant GPU compute resources wasted. We present NEO, an online LLM inference system that offloads part of attention compute and KV cache states from the GPU to the local host CPU, effectively increasing the GPU batch size and thus inference throughput. To this end, NEO proposes asymmetric GPU-CPU pipelining and load-aware scheduling to balance GPU and CPU loads and fully utilize their compute and memory resources. We evaluate NEO on a wide range of workloads (i.e., code generation, text summarization), GPUs (i.e., T4, A10G, H100), and LLM models (i.e., 7B, 8B, 70B). NEO achieves up to 7.5$\times$, 26%, and 14% higher throughput compared to GPU-only approach on T4, A10G, and H100 GPUs, respectively, while maintaining the same latency; with more powerful CPUs, NEO achieves up to 79.3% throughput gain on A10G GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01136v1">Do LLMs Know to Respect Copyright Notice?</a></div>
    <div class="paper-meta">
      📅 2024-11-02
      | 💬 EMNLP 2024 main
    </div>
    <details class="paper-abstract">
      Prior study shows that LLMs sometimes generate content that violates copyright. In this paper, we study another important yet underexplored problem, i.e., will LLMs respect copyright information in user input, and behave accordingly? The research problem is critical, as a negative answer would imply that LLMs will become the primary facilitator and accelerator of copyright infringement behavior. We conducted a series of experiments using a diverse set of language models, user prompts, and copyrighted materials, including books, news articles, API documentation, and movie scripts. Our study offers a conservative evaluation of the extent to which language models may infringe upon copyrights when processing user input containing protected material. This research emphasizes the need for further investigation and the importance of ensuring LLMs respect copyright regulations when handling user input to prevent unauthorized use or reproduction of protected content. We also release a benchmark dataset serving as a test bed for evaluating infringement behaviors by LLMs and stress the need for future alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07018v2">Tri-Level Navigator: LLM-Empowered Tri-Level Learning for Time Series OOD Generalization</a></div>
    <div class="paper-meta">
      📅 2024-11-02
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Out-of-Distribution (OOD) generalization in machine learning is a burgeoning area of study. Its primary goal is to enhance the adaptability and resilience of machine learning models when faced with new, unseen, and potentially adversarial data that significantly diverges from their original training datasets. In this paper, we investigate time series OOD generalization via pre-trained Large Language Models (LLMs). We first propose a novel \textbf{T}ri-level learning framework for \textbf{T}ime \textbf{S}eries \textbf{O}OD generalization, termed TTSO, which considers both sample-level and group-level uncertainties. This formula offers a fresh theoretic perspective for formulating and analyzing OOD generalization problem. In addition, we provide a theoretical analysis to justify this method is well motivated. We then develop a stratified localization algorithm tailored for this tri-level optimization problem, theoretically demonstrating the guaranteed convergence of the proposed algorithm. Our analysis also reveals that the iteration complexity to obtain an $\epsilon$-stationary point is bounded by O($\frac{1}{\epsilon^{2}}$). Extensive experiments on real-world datasets have been conducted to elucidate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03343v1">What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      While `jailbreaks' have been central to research on the safety and reliability of LLMs (large language models), the underlying mechanisms behind these attacks are not well understood. Some prior works have used linear methods to analyze jailbreak prompts or model refusal. Here, however, we compare linear and nonlinear methods to study the features in prompts that contribute to successful jailbreaks. We do this by probing for jailbreak success based only on the portions of the latent representations corresponding to prompt tokens. First, we introduce a dataset of 10,800 jailbreak attempts from 35 attack methods. We then show that different jailbreaking methods work via different nonlinear features in prompts. Specifically, we find that while probes can distinguish between successful and unsuccessful jailbreaking prompts with a high degree of accuracy, they often transfer poorly to held-out attack methods. We also show that nonlinear probes can be used to mechanistically jailbreak the LLM by guiding the design of adversarial latent perturbations. These mechanistic jailbreaks are able to jailbreak Gemma-7B-IT more reliably than 34 of the 35 techniques that it was trained on. Ultimately, our results suggest that jailbreaks cannot be thoroughly understood in terms of universal or linear prompt features alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02442v1">TODO: Enhancing LLM Alignment with Ternary Preferences</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human intent is critical for enhancing their performance across a variety of tasks. Standard alignment techniques, such as Direct Preference Optimization (DPO), often rely on the binary Bradley-Terry (BT) model, which can struggle to capture the complexities of human preferences -- particularly in the presence of noisy or inconsistent labels and frequent ties. To address these limitations, we introduce the Tie-rank Oriented Bradley-Terry model (TOBT), an extension of the BT model that explicitly incorporates ties, enabling more nuanced preference representation. Building on this, we propose Tie-rank Oriented Direct Preference Optimization (TODO), a novel alignment algorithm that leverages TOBT's ternary ranking system to improve preference alignment. In evaluations on Mistral-7B and Llama 3-8B models, TODO consistently outperforms DPO in modeling preferences across both in-distribution and out-of-distribution datasets. Additional assessments using MT Bench and benchmarks such as Piqa, ARC-c, and MMLU further demonstrate TODO's superior alignment performance. Notably, TODO also shows strong results in binary preference alignment, highlighting its versatility and potential for broader integration into LLM alignment. The implementation details can be found in https://github.com/XXares/TODO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01245v1">PMoL: Parameter Efficient MoE for Preference Mixing of LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) has been proven to be an effective method for preference alignment of large language models (LLMs) and is widely used in the post-training process of LLMs. However, RLHF struggles with handling multiple competing preferences. This leads to a decrease in the alignment of LLMs with human preferences. To address this issue, we propose Preference Mixture of LoRAs (PMoL) from the perspective of model architecture, which can adapt to any number of preferences to mix. PMoL combines Mixture of Experts (MoE) and Low Rank Adaptor (LoRA). This architecture is innovatively applied to the research of preference alignment and has achieved significant performance improvement. The expert group soft loss is used to enable MoE with the ability to mix preferences. Through comprehensive evaluation by the reward model and GPT-4o, the experiment results show that PMoL has superior preference mixing capabilities compared to baseline methods. PMoL achieves better preference alignment with lower training costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01213v1">One Arrow, Many Targets: Probing LLMs for Multi-Attribute Controllable Text Summarization</a></div>
    <div class="paper-meta">
      📅 2024-11-02
    </div>
    <details class="paper-abstract">
      Text summarization is a well-established task within the natural language processing (NLP) community. However, the focus on controllable summarization tailored to user requirements is gaining traction only recently. While several efforts explore controllability in text summarization, the investigation of Multi-Attribute Controllable Summarization (MACS) remains limited. This work addresses this gap by examining the MACS task through the lens of large language models (LLMs), using various learning paradigms, particularly low-rank adapters. We experiment with different popular adapter fine-tuning strategies to assess the effectiveness of the resulting models in retaining cues and patterns associated with multiple controllable attributes. Additionally, we propose and evaluate a novel hierarchical adapter fusion technique to integrate learnings from two distinct controllable attributes. Subsquently, we present our findings, discuss the challenges encountered, and suggest potential avenues for advancing the MACS task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09114v2">Catastrophic Cyber Capabilities Benchmark (3CB): Robustly Evaluating LLM Agent Cyber Offense Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-11-02
      | 💬 https://cybercapabilities.org/
    </div>
    <details class="paper-abstract">
      LLM agents have the potential to revolutionize defensive cyber operations, but their offensive capabilities are not yet fully understood. To prepare for emerging threats, model developers and governments are evaluating the cyber capabilities of foundation models. However, these assessments often lack transparency and a comprehensive focus on offensive capabilities. In response, we introduce the Catastrophic Cyber Capabilities Benchmark (3CB), a novel framework designed to rigorously assess the real-world offensive capabilities of LLM agents. Our evaluation of modern LLMs on 3CB reveals that frontier models, such as GPT-4o and Claude 3.5 Sonnet, can perform offensive tasks such as reconnaissance and exploitation across domains ranging from binary analysis to web technologies. Conversely, smaller open-source models exhibit limited offensive capabilities. Our software solution and the corresponding benchmark provides a critical tool to reduce the gap between rapidly improving capabilities and robustness of cyber offense evaluations, aiding in the safer deployment and regulation of these powerful technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13006v2">LLM Chain Ensembles for Scalable and Accurate Data Annotation</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      The ability of large language models (LLMs) to perform zero-shot classification makes them viable solutions for data annotation in rapidly evolving domains where quality labeled data is often scarce and costly to obtain. However, the large-scale deployment of LLMs can be prohibitively expensive. This paper introduces an LLM chain ensemble methodology that aligns multiple LLMs in a sequence, routing data subsets to subsequent models based on classification uncertainty. This approach leverages the strengths of individual LLMs within a broader system, allowing each model to handle data points where it exhibits the highest confidence, while forwarding more complex cases to potentially more robust models. Our results show that the chain ensemble method often exceeds the performance of the best individual model in the chain and achieves substantial cost savings, making LLM chain ensembles a practical and efficient solution for large-scale data annotation challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08217v2">RED-CT: A Systems Design Methodology for Using LLM-labeled Data to Train and Deploy Edge Classifiers for Computational Social Science</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enhanced our ability to rapidly analyze and classify unstructured natural language data. However, concerns regarding cost, network limitations, and security constraints have posed challenges for their integration into work processes. In this study, we adopt a systems design approach to employing LLMs as imperfect data annotators for downstream supervised learning tasks, introducing novel system intervention measures aimed at improving classification performance. Our methodology outperforms LLM-generated labels in seven of eight tests, demonstrating an effective strategy for incorporating LLMs into the design and deployment of specialized, supervised learning models present in many industry use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13047v2">LLM Confidence Evaluation Measures in Zero-Shot CSS Classification</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Assessing classification confidence is critical for leveraging large language models (LLMs) in automated labeling tasks, especially in the sensitive domains presented by Computational Social Science (CSS) tasks. In this paper, we make three key contributions: (1) we propose an uncertainty quantification (UQ) performance measure tailored for data annotation tasks, (2) we compare, for the first time, five different UQ strategies across three distinct LLMs and CSS data annotation tasks, (3) we introduce a novel UQ aggregation strategy that effectively identifies low-confidence LLM annotations and disproportionately uncovers data incorrectly labeled by the LLMs. Our results demonstrate that our proposed UQ aggregation strategy improves upon existing methods andcan be used to significantly improve human-in-the-loop data annotation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v1">Emoji Attack: A Method for Misleading Judge LLMs in Safety Risk Detection</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Jailbreaking attacks show how Large Language Models (LLMs) can be tricked into generating harmful outputs using malicious prompts. To prevent these attacks, other LLMs are often used as judges to evaluate the harmfulness of the generated content. However, relying on LLMs as judges can introduce biases into the detection process, which in turn compromises the effectiveness of the evaluation. In this paper, we show that Judge LLMs, like other LLMs, are also affected by token segmentation bias. This bias occurs when tokens are split into smaller sub-tokens, altering their embeddings. This makes it harder for the model to detect harmful content. Specifically, this bias can cause sub-tokens to differ significantly from the original token in the embedding space, leading to incorrect "safe" predictions for harmful content. To exploit this bias in Judge LLMs, we introduce the Emoji Attack -- a method that places emojis within tokens to increase the embedding differences between sub-tokens and their originals. These emojis create new tokens that further distort the token embeddings, exacerbating the bias. To counter the Emoji Attack, we design prompts that help LLMs filter out unusual characters. However, this defense can still be bypassed by using a mix of emojis and other characters. The Emoji Attack can also be combined with existing jailbreaking prompts using few-shot learning, which enables LLMs to generate harmful responses with emojis. These responses are often mistakenly labeled as "safe" by Judge LLMs, allowing the attack to slip through. Our experiments with six state-of-the-art Judge LLMs show that the Emoji Attack allows 25\% of harmful responses to bypass detection by Llama Guard and Llama Guard 2, and up to 75\% by ShieldLM. These results highlight the need for stronger Judge LLMs to address this vulnerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01073v1">AttackQA: Development and Adoption of a Dataset for Assisting Cybersecurity Operations using Fine-tuned and Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) on specialized domain datasets has shown improved performance when large language models (LLMs) are fine-tuned for generating responses to user queries. In this study, we develop a cybersecurity question-answering (Q\&A) dataset, called AttackQA, and employ it to build a RAG-based Q\&A system designed for analysts in security operations centers. The dataset comprises 25,335 Q\&A pairs, accompanied by rationales to facilitate fine-tuning and evaluation. 80\% of the dataset was generated with help of a lightweight open-source LLM (LLama 3 8B), which produced over 1100 tokens per second with full 16-bit precision on SambaNova System's SN40L specialized hardware. To ensure dataset quality, we fine-tuned LLama 3 70B to detect and reject low-quality Q\&A pairs. In using the dataset for RAG, we demonstrate that fine-tuning open-source embeddings and LLMs can yield superior accuracy compared to OpenAI's state-of-the-art proprietary embedding and LLM (GPT-4o). Furthermore, we use Llama 3.1 405B as a judge to evaluate answer correctness, enabling the creation of a fully open-source, high-speed RAG and evaluation pipeline with a benchmark for model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05215v2">DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) greatly improves model quality for downstream tasks. However, serving many fine-tuned LLMs concurrently is challenging due to the sporadic, bursty, and varying request patterns of different LLMs. To bridge this gap, we present DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by up to 10x while maintaining high model quality. The key insight behind this design is that fine-tuning results in small-magnitude changes to the pre-trained model. By co-designing the serving system with the compression algorithm, DeltaZip achieves 2x to 12x improvement in throughput compared to the state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02502v2">Context Conquers Parameters: Outperforming Proprietary LLM in Commit Message Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Accepted in ICSE 2025
    </div>
    <details class="paper-abstract">
      Commit messages provide descriptions of the modifications made in a commit using natural language, making them crucial for software maintenance and evolution. Recent developments in Large Language Models (LLMs) have led to their use in generating high-quality commit messages, such as the Omniscient Message Generator (OMG). This method employs GPT-4 to produce state-of-the-art commit messages. However, the use of proprietary LLMs like GPT-4 in coding tasks raises privacy and sustainability concerns, which may hinder their industrial adoption. Considering that open-source LLMs have achieved competitive performance in developer tasks such as compiler validation, this study investigates whether they can be used to generate commit messages that are comparable with OMG. Our experiments show that an open-source LLM can generate commit messages that are comparable to those produced by OMG. In addition, through a series of contextual refinements, we propose lOcal MessagE GenerAtor (OMEGA) , a CMG approach that uses a 4-bit quantized 8B open-source LLM. OMEGA produces state-of-the-art commit messages, surpassing the performance of GPT-4 in practitioners' preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01022v1">Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM Generation Output</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 To appear in Proceedings of EMNLP 2024 Industry Track
    </div>
    <details class="paper-abstract">
      We present a light-weight approach for detecting nonfactual outputs from retrieval-augmented generation (RAG). Given a context and putative output, we compute a factuality score that can be thresholded to yield a binary decision to check the results of LLM-based question-answering, summarization, or other systems. Unlike factuality checkers that themselves rely on LLMs, we use compact, open-source natural language inference (NLI) models that yield a freely accessible solution with low latency and low cost at run-time, and no need for LLM fine-tuning. The approach also enables downstream mitigation and correction of hallucinations, by tracing them back to specific context chunks. Our experiments show high area under the ROC curve (AUC) across a wide range of relevant open source datasets, indicating the effectiveness of our method for fact-checking RAG output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04182v3">Regression-aware Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong results on a range of applications, including regression and scoring tasks. Typically, one obtains outputs from an LLM via autoregressive sampling from the model's output distribution. We show that this inference strategy can be sub-optimal for common regression and scoring evaluation metrics. As a remedy, we build on prior work on Minimum Bayes Risk decoding, and propose alternate inference strategies that estimate the Bayes-optimal solution for regression and scoring metrics in closed-form from sampled responses. We show that our proposal significantly improves over baselines across datasets and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11856v1">Can EDA Tool Feedback Improve Verilog Generation by LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Traditionally, digital hardware designs are written in the Verilog hardware description language (HDL) and debugged manually by engineers. This can be time-consuming and error-prone for complex designs. Large Language Models (LLMs) are emerging as a potential tool to help generate fully functioning HDL code, but most works have focused on generation in the single-shot capacity: i.e., run and evaluate, a process that does not leverage debugging and as such does not adequately reflect a realistic development process. In this work we evaluate the ability of LLMs to leverage feedback from electronic design automation (EDA) tools to fix mistakes in their own generated Verilog. To accomplish this we present an open-source, highly customizable framework, AutoChip, which combines conversational LLMs with the output from Verilog compilers and simulations to iteratively generate and repair Verilog. To determine the success of these LLMs we leverage the VerilogEval benchmark set. We evaluate four state-of-the-art conversational LLMs, focusing on readily accessible commercial models. EDA tool feedback proved to be consistently more effective than zero-shot prompting only with GPT-4o, the most computationally complex model we evaluated. In the best case we observed a 5.8% increase in the number of successful designs with a 34.2% decrease in cost over the best zero-shot results. Mixing smaller models with this larger model at the end of the feedback iterations resulted in equally as much success as with GPT-4o using feedback, but for an additional 41.9% less cost (overall decrease in cost over zero-shot of 89.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00750v1">Mitigating Tail Narrowing in LLM Self-Improvement via Socratic-Guided Sampling</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Codes are publicly available at https://github.com/Yiwen-Ding/Guided-Self-Improvement
    </div>
    <details class="paper-abstract">
      Self-improvement methods enable large language models (LLMs) to generate solutions themselves and iteratively train on filtered, high-quality rationales. This process proves effective and reduces the reliance on human supervision in LLMs' reasoning, but the performance soon plateaus. We delve into the process and find that models tend to over-sample on easy queries and under-sample on queries they have yet to master. As iterations proceed, this imbalance in sampling is exacerbated, leading to a long-tail distribution where solutions to difficult queries almost diminish. This phenomenon limits the performance gain of self-improving models. A straightforward solution is brute-force sampling to balance the distribution, which significantly raises computational costs. In this paper, we introduce Guided Self-Improvement (GSI), a strategy aimed at improving the efficiency of sampling challenging heavy-tailed data. It leverages Socratic-style guidance signals to help LLM reasoning with complex queries, reducing the exploration effort and minimizing computational overhead. Experiments on four models across diverse mathematical tasks show that GSI strikes a balance between performance and efficiency, while also being effective on held-out tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00932v1">LLMs: A Game-Changer for Software Engineers?</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 20 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like GPT-3 and GPT-4 have emerged as groundbreaking innovations with capabilities that extend far beyond traditional AI applications. These sophisticated models, trained on massive datasets, can generate human-like text, respond to complex queries, and even write and interpret code. Their potential to revolutionize software development has captivated the software engineering (SE) community, sparking debates about their transformative impact. Through a critical analysis of technical strengths, limitations, real-world case studies, and future research directions, this paper argues that LLMs are not just reshaping how software is developed but are redefining the role of developers. While challenges persist, LLMs offer unprecedented opportunities for innovation and collaboration. Early adoption of LLMs in software engineering is crucial to stay competitive in this rapidly evolving landscape. This paper serves as a guide, helping developers, organizations, and researchers understand how to harness the power of LLMs to streamline workflows and acquire the necessary skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01721v3">DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 NeurIPS 2024 Oral, Website at https://duquant.github.io
    </div>
    <details class="paper-abstract">
      Quantization of large language models (LLMs) faces significant challenges, particularly due to the presence of outlier activations that impede efficient low-bit representation. Traditional approaches predominantly address Normal Outliers, which are activations across all tokens with relatively large magnitudes. However, these methods struggle with smoothing Massive Outliers that display significantly larger values, which leads to significant performance degradation in low-bit quantization. In this paper, we introduce DuQuant, a novel approach that utilizes rotation and permutation transformations to more effectively mitigate both massive and normal outliers. First, DuQuant starts by constructing the rotation matrix, using specific outlier dimensions as prior knowledge, to redistribute outliers to adjacent channels by block-wise rotation. Second, We further employ a zigzag permutation to balance the distribution of outliers across blocks, thereby reducing block-wise variance. A subsequent rotation further smooths the activation landscape, enhancing model performance. DuQuant simplifies the quantization process and excels in managing outliers, outperforming the state-of-the-art baselines across various sizes and types of LLMs on multiple tasks, even with 4-bit weight-activation quantization. Our code is available at https://github.com/Hsu1023/DuQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15589v3">Efficient Adversarial Training in LLMs with Continuous Attacks</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 19 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02432v1">Can LLMs make trade-offs involving stipulated pain and pleasure states?</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Pleasure and pain play an important role in human decision making by providing a common currency for resolving motivational conflicts. While Large Language Models (LLMs) can generate detailed descriptions of pleasure and pain experiences, it is an open question whether LLMs can recreate the motivational force of pleasure and pain in choice scenarios - a question which may bear on debates about LLM sentience, understood as the capacity for valenced experiential states. We probed this question using a simple game in which the stated goal is to maximise points, but where either the points-maximising option is said to incur a pain penalty or a non-points-maximising option is said to incur a pleasure reward, providing incentives to deviate from points-maximising behaviour. Varying the intensity of the pain penalties and pleasure rewards, we found that Claude 3.5 Sonnet, Command R+, GPT-4o, and GPT-4o mini each demonstrated at least one trade-off in which the majority of responses switched from points-maximisation to pain-minimisation or pleasure-maximisation after a critical threshold of stipulated pain or pleasure intensity is reached. LLaMa 3.1-405b demonstrated some graded sensitivity to stipulated pleasure rewards and pain penalties. Gemini 1.5 Pro and PaLM 2 prioritised pain-avoidance over points-maximisation regardless of intensity, while tending to prioritise points over pleasure regardless of intensity. We discuss the implications of these findings for debates about the possibility of LLM sentience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00642v1">LLM-Based Misconfiguration Detection for AWS Serverless Computing</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Serverless computing is an emerging cloud computing paradigm that enables developers to build applications at the function level, known as serverless applications. Amazon Web Services (AWS), the leading provider in this domain, provides the Serverless Application Model (AWS SAM), the most widely adopted configuration schema for configuring and managing serverless applications through a specified file. However, misconfigurations pose a significant challenge in serverless development. Traditional data-driven techniques may struggle with serverless applications because the complexity of serverless configurations hinders pattern recognition, and it is challenging to gather complete datasets that cover all possible configurations. Leveraging vast amounts of publicly available data during pre-training, LLMs can have the potential to assist in identifying and explaining misconfigurations in serverless applications. In this paper, we introduce SlsDetector, the first framework leveraging LLMs to detect misconfigurations in serverless applications. SlsDetector utilizes effective prompt engineering with zero-shot learning to identify configuration issues. It designs multi-dimensional constraints specifically tailored to the configuration characteristics of serverless applications and leverages the Chain of Thought technique to enhance LLMs inferences. We evaluate SlsDetector on a curated dataset of 110 configuration files. Our results show that SlsDetector, based on ChatGPT-4o, achieves a precision of 72.88%, recall of 88.18%, and F1-score of 79.75%, outperforming state-of-the-art data-driven approaches by 53.82, 17.40, and 49.72 percentage points, respectively. Furthermore, we investigate the generalization capability of SlsDetector by applying recent LLMs, including Llama 3.1 (405B) Instruct Turbo and Gemini 1.5 Pro, with results showing consistently high effectiveness across these models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15420v3">XC-Cache: Cross-Attending to Cached Context for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) approaches typically leverage prompting to condition decoder-only language model generation on reference information. Just-in-time processing of a context is inefficient due to the quadratic cost of self-attention operations, and caching is desirable. However, caching transformer states can easily require almost as much space as the model parameters. When the right context isn't known in advance, caching ICL can be challenging. This work addresses these limitations by introducing models that, inspired by the encoder-decoder architecture, use cross-attention to condition generation on reference text without the prompt. More precisely, we leverage pre-trained decoder-only models and only train a small number of added layers. We use Question-Answering (QA) as a testbed to evaluate the ability of our models to perform conditional generation and observe that they outperform ICL, are comparable to fine-tuned prompted LLMs, and drastically reduce the space footprint relative to standard KV caching by two orders of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20778v2">Improved Generation of Adversarial Examples Against Safety-aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing automatic jailbreak attacks against safety-aligned LLMs. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, into gradient-based adversarial prompt generation and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that 87% of the query-specific adversarial suffixes generated by the developed combination can induce Llama-2-7B-Chat to produce the output that exactly matches the target string on AdvBench. This match rate is 33% higher than that of a very strong baseline known as GCG, demonstrating advanced discrete optimization for adversarial prompt generation against LLMs. In addition, without introducing obvious cost, the combination achieves >30% absolute increase in attack success rates compared with GCG when generating both query-specific (38% -> 68%) and universal adversarial prompts (26.68% -> 60.32%) for attacking the Llama-2-7B-Chat model on AdvBench. Code at: https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00418v1">Self-Evolved Reward Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 19 pages,6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. These methods can be costly and may introduce biases that affect the language model's responses. As language models improve, human input may become less effective in further enhancing their performance. In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v1">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 26 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but often produce hallucinations for complex ones. While integrating LLMs with tools can increase reliability, this approach typically results in over-reliance on tools, diminishing the model's ability to solve simple problems through basic reasoning. In contrast, human experts first assess problem complexity using domain knowledge before choosing an appropriate solution approach. Inspired by this human problem-solving process, we propose a novel two-component fine-tuning method. In the first component World Knowledge Distillation (WKD), LLMs learn directly from solutions generated using tool's information to internalize domain knowledge. In the second component Tool Usage Adaptation (TUA), we partition problems into easy and hard categories based on the model's direct answering accuracy. While maintaining the same alignment target for easy problems as in WKD, we train the model to intelligently switch to tool usage for more challenging problems. We validate our method on six scientific benchmark datasets, spanning mathematics, climate science and epidemiology. On average, our models demonstrate a 28.18% improvement in answer accuracy and a 13.89% increase in tool usage precision across all datasets, surpassing state-of-the-art models including GPT-4o and Claude-3.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00348v1">Attention Tracker: Detecting Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00331v1">Beyond Utility: Evaluating LLM as Recommender</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      With the rapid development of Large Language Models (LLMs), recent studies employed LLMs as recommenders to provide personalized information services for distinct users. Despite efforts to improve the accuracy of LLM-based recommendation models, relatively little attention is paid to beyond-utility dimensions. Moreover, there are unique evaluation aspects of LLM-based recommendation models, which have been largely ignored. To bridge this gap, we explore four new evaluation dimensions and propose a multidimensional evaluation framework. The new evaluation dimensions include: 1) history length sensitivity, 2) candidate position bias, 3) generation-involved performance, and 4) hallucinations. All four dimensions have the potential to impact performance, but are largely unnecessary for consideration in traditional systems. Using this multidimensional evaluation framework, along with traditional aspects, we evaluate the performance of seven LLM-based recommenders, with three prompting strategies, comparing them with six traditional models on both ranking and re-ranking tasks on four datasets. We find that LLMs excel at handling tasks with prior knowledge and shorter input histories in the ranking setting, and perform better in the re-ranking setting, beating traditional models across multiple dimensions. However, LLMs exhibit substantial candidate position bias issues, and some models hallucinate non-existent items much more often than others. We intend our evaluation framework and observations to benefit future research on the use of LLMs as recommenders. The code and data are available at https://github.com/JiangDeccc/EvaLLMasRecommender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15880v2">HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Many structured prediction and reasoning tasks can be framed as program synthesis problems, where the goal is to generate a program in a domain-specific language (DSL) that transforms input data into the desired output. Unfortunately, purely neural approaches, such as large language models (LLMs), often fail to produce fully correct programs in unfamiliar DSLs, while purely symbolic methods based on combinatorial search scale poorly to complex problems. Motivated by these limitations, we introduce a hybrid approach, where LLM completions for a given task are used to learn a task-specific, context-free surrogate model, which is then used to guide program synthesis. We evaluate this hybrid approach on three domains, and show that it outperforms both unguided search and direct sampling from LLMs, as well as existing program synthesizers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08357v2">An Experimental Study of Competitive Market Behavior Through LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      This study explores the potential of large language models (LLMs) to conduct market experiments, aiming to understand their capability to comprehend competitive market dynamics. We model the behavior of market agents in a controlled experimental setting, assessing their ability to converge toward competitive equilibria. The results reveal the challenges current LLMs face in replicating the dynamic decision-making processes characteristic of human trading behavior. Unlike humans, LLMs lacked the capacity to achieve market equilibrium. The research demonstrates that while LLMs provide a valuable tool for scalable and reproducible market simulations, their current limitations necessitate further advancements to fully capture the complexities of market behavior. Future work that enhances dynamic learning capabilities and incorporates elements of behavioral economics could improve the effectiveness of LLMs in the economic domain, providing new insights into market dynamics and aiding in the refinement of economic policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14556v2">RACOON: An LLM-based Framework for Retrieval-Augmented Column Type Annotation with a Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      As an important component of data exploration and integration, Column Type Annotation (CTA) aims to label columns of a table with one or more semantic types. With the recent development of Large Language Models (LLMs), researchers have started to explore the possibility of using LLMs for CTA, leveraging their strong zero-shot capabilities. In this paper, we build on this promising work and improve on LLM-based methods for CTA by showing how to use a Knowledge Graph (KG) to augment the context information provided to the LLM. Our approach, called RACOON, combines both pre-trained parametric and non-parametric knowledge during generation to improve LLMs' performance on CTA. Our experiments show that RACOON achieves up to a 0.21 micro F-1 improvement compared against vanilla LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05972v2">Decision-Making Behavior Evaluation Framework for LLMs under Uncertain Context</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 Jingru Jia and Zehua Yuan have equal contribution
    </div>
    <details class="paper-abstract">
      When making decisions under uncertainty, individuals often deviate from rational behavior, which can be evaluated across three dimensions: risk preference, probability weighting, and loss aversion. Given the widespread use of large language models (LLMs) in decision-making processes, it is crucial to assess whether their behavior aligns with human norms and ethical expectations or exhibits potential biases. Several empirical studies have investigated the rationality and social behavior performance of LLMs, yet their internal decision-making tendencies and capabilities remain inadequately understood. This paper proposes a framework, grounded in behavioral economics, to evaluate the decision-making behaviors of LLMs. Through a multiple-choice-list experiment, we estimate the degree of risk preference, probability weighting, and loss aversion in a context-free setting for three commercial LLMs: ChatGPT-4.0-Turbo, Claude-3-Opus, and Gemini-1.0-pro. Our results reveal that LLMs generally exhibit patterns similar to humans, such as risk aversion and loss aversion, with a tendency to overweight small probabilities. However, there are significant variations in the degree to which these behaviors are expressed across different LLMs. We also explore their behavior when embedded with socio-demographic features, uncovering significant disparities. For instance, when modeled with attributes of sexual minority groups or physical disabilities, Claude-3-Opus displays increased risk aversion, leading to more conservative choices. These findings underscore the need for careful consideration of the ethical implications and potential biases in deploying LLMs in decision-making scenarios. Therefore, this study advocates for developing standards and guidelines to ensure that LLMs operate within ethical boundaries while enhancing their utility in complex decision-making environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10455v1">LLM-itation is the Sincerest Form of Data: Generating Synthetic Buggy Code Submissions for Computing Education</a></div>
    <div class="paper-meta">
      📅 2024-11-01
      | 💬 7 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      There is a great need for data in computing education research. Data is needed to understand how students behave, to train models of student behavior to optimally support students, and to develop and validate new assessment tools and learning analytics techniques. However, relatively few computing education datasets are shared openly, often due to privacy regulations and issues in making sure the data is anonymous. Large language models (LLMs) offer a promising approach to create large-scale, privacy-preserving synthetic data, which can be used to explore various aspects of student learning, develop and test educational technologies, and support research in areas where collecting real student data may be challenging or impractical. This work explores generating synthetic buggy code submissions for introductory programming exercises using GPT-4o. We compare the distribution of test case failures between synthetic and real student data from two courses to analyze the accuracy of the synthetic data in mimicking real student data. Our findings suggest that LLMs can be used to generate synthetic incorrect submissions that are not significantly different from real student data with regard to test case failure distributions. Our research contributes to the development of reliable synthetic datasets for computing education research and teaching, potentially accelerating progress in the field while preserving student privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16218v2">Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      We study a class of optimization problems motivated by automating the design and update of AI systems like coding assistants, robots, and copilots. AutoDiff frameworks, like PyTorch, enable efficient end-to-end optimization of differentiable systems. However, general computational workflows can be non-differentiable and involve rich feedback (e.g. console output or user's responses), heterogeneous parameters (e.g. prompts, codes), and intricate objectives (beyond maximizing a score). We investigate end-to-end generative optimization -- using generative models such as LLMs within the optimizer for automatic updating of general computational workflows. We discover that workflow execution traces are akin to back-propagated gradients in AutoDiff and can provide key information to interpret feedback for efficient optimization. Formally, we frame a new mathematical setup, Optimization with Trace Oracle (OPTO). In OPTO, an optimizer receives an execution trace along with feedback on the computed output and updates parameters iteratively. We provide a Python library, Trace, that efficiently converts a workflow optimization problem into an OPTO instance using PyTorch-like syntax. Using Trace, we develop a general LLM-based generative optimizer called OptoPrime. In empirical studies, we find that OptoPrime is capable of first-order numerical optimization, prompt optimization, hyper-parameter tuning, robot controller design, code debugging, etc., and is often competitive with specialized optimizers for each domain. We envision Trace as an open research platform for devising novel generative optimizers and developing the next generation of interactive learning agents. Website: https://microsoft.github.io/Trace/.
    </details>
</div>
