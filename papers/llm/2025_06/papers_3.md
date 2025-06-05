# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10918v2">INVARLLM: LLM-assisted Physical Invariant Extraction for Cyber-Physical Systems Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Cyber-Physical Systems (CPS) are vulnerable to cyber-physical attacks that violate physical laws. While invariant-based anomaly detection is effective, existing methods are limited: data-driven approaches lack semantic context, and physics-based models require extensive manual work. We propose INVARLLM, a hybrid framework that uses large language models (LLMs) to extract semantic information from CPS documentation and generate physical invariants, then validates these against real system data using a PCMCI+-inspired K-means method. This approach combines LLM semantic understanding with empirical validation to ensure both interpretability and reliability. We evaluate INVARLLM on SWaT and WADI datasets, achieving 100% precision in anomaly detection with no false alarms, outperforming all existing methods. Our results demonstrate that integrating LLM-derived semantics with statistical validation provides a scalable and dependable solution for CPS security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01839v1">Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from static tools to fully agentic systems, their potential for transforming social science research has become increasingly evident. This paper introduces a structured framework for understanding the diverse applications of LLM-based agents, ranging from simple data processors to complex, multi-agent systems capable of simulating emergent social dynamics. By mapping this developmental continuum across six levels, the paper clarifies the technical and methodological boundaries between different agentic architectures, providing a comprehensive overview of current capabilities and future potential. It highlights how lower-tier systems streamline conventional tasks like text classification and data annotation, while higher-tier systems enable novel forms of inquiry, including the study of group dynamics, norm formation, and large-scale social processes. However, these advancements also introduce significant challenges, including issues of reproducibility, ethical oversight, and the risk of emergent biases. The paper critically examines these concerns, emphasizing the need for robust validation protocols, interdisciplinary collaboration, and standardized evaluation metrics. It argues that while LLM-based agents hold transformative potential for the social sciences, realizing this promise will require careful, context-sensitive deployment and ongoing methodological refinement. The paper concludes with a call for future research that balances technical innovation with ethical responsibility, encouraging the development of agentic systems that not only replicate but also extend the frontiers of social science, offering new insights into the complexities of human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01825v1">Which Factors Make Code LLMs More Vulnerable to Backdoor Attacks? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Code LLMs are increasingly employed in software development. However, studies have shown that they are vulnerable to backdoor attacks: when a trigger (a specific input pattern) appears in the input, the backdoor will be activated and cause the model to generate malicious outputs. Researchers have designed various triggers and demonstrated the feasibility of implanting backdoors by poisoning a fraction of the training data. Some basic conclusions have been made, such as backdoors becoming easier to implant when more training data are modified. However, existing research has not explored other factors influencing backdoor attacks on Code LLMs, such as training batch size, epoch number, and the broader design space for triggers, e.g., trigger length. To bridge this gap, we use code summarization as an example to perform an empirical study that systematically investigates the factors affecting backdoor effectiveness and understands the extent of the threat posed. Three categories of factors are considered: data, model, and inference, revealing previously overlooked findings. We find that the prevailing consensus -- that attacks are ineffective at extremely low poisoning rates -- is incorrect. The absolute number of poisoned samples matters as well. Specifically, poisoning just 20 out of 454K samples (0.004\% poisoning rate -- far below the minimum setting of 0.1\% in prior studies) successfully implants backdoors! Moreover, the common defense is incapable of removing even a single poisoned sample from it. Additionally, small batch sizes increase the risk of backdoor attacks. We also uncover other critical factors such as trigger types, trigger length, and the rarity of tokens in the triggers, leading to valuable insights for assessing Code LLMs' vulnerability to backdoor attacks. Our study highlights the urgent need for defense mechanisms against extremely low poisoning rate settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01814v1">Analysis of LLM Bias (Chinese Propaganda & Anti-US Sentiment) in DeepSeek-R1 vs. ChatGPT o3-mini-high</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly shape public understanding and civic decisions, yet their ideological neutrality is a growing concern. While existing research has explored various forms of LLM bias, a direct, cross-lingual comparison of models with differing geopolitical alignments-specifically a PRC-system model versus a non-PRC counterpart-has been lacking. This study addresses this gap by systematically evaluating DeepSeek-R1 (PRC-aligned) against ChatGPT o3-mini-high (non-PRC) for Chinese-state propaganda and anti-U.S. sentiment. We developed a novel corpus of 1,200 de-contextualized, reasoning-oriented questions derived from Chinese-language news, presented in Simplified Chinese, Traditional Chinese, and English. Answers from both models (7,200 total) were assessed using a hybrid evaluation pipeline combining rubric-guided GPT-4o scoring with human annotation. Our findings reveal significant model-level and language-dependent biases. DeepSeek-R1 consistently exhibited substantially higher proportions of both propaganda and anti-U.S. bias compared to ChatGPT o3-mini-high, which remained largely free of anti-U.S. sentiment and showed lower propaganda levels. For DeepSeek-R1, Simplified Chinese queries elicited the highest bias rates; these diminished in Traditional Chinese and were nearly absent in English. Notably, DeepSeek-R1 occasionally responded in Simplified Chinese to Traditional Chinese queries and amplified existing PRC-aligned terms in its Chinese answers, demonstrating an "invisible loudspeaker" effect. Furthermore, such biases were not confined to overtly political topics but also permeated cultural and lifestyle content, particularly in DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01804v1">A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01770v1">ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at https://github.com/weizeming/ReGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01734v1">Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law -- a statistical pattern where lower digits occur more frequently as leading digits -- we hypothesize that the long-tailed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark with uniformly distributed ground-truth digits across seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01732v1">Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. These data most often contain trillions of tokens with large portions of copyrighted or proprietary content, which hinders the usage of such models under AI legislation. This raises the need for truly open pre-training data that is compliant with the data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for language model pre-training. The data assembled in Common Corpus are either uncopyrighted or under permissible licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the main European languages to low-resource ones rarely present in pre-training datasets; in addition, it includes a large portion of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs in diverse areas of knowledge. In this technical report, we present the detailed provenance of data assembling and the details of dataset filtering and curation. Being already used by such industry leaders as Anthropic and multiple LLM training projects, we believe that Common Corpus will become a critical infrastructure for open science research in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v1">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01702v1">mdok of KInIT: Robustly Fine-tuned LLM for Binary and Multiclass AI-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) are able to generate high-quality texts in multiple languages. Such texts are often not recognizable by humans as generated, and therefore present a potential of LLMs for misuse (e.g., plagiarism, spams, disinformation spreading). An automated detection is able to assist humans to indicate the machine-generated texts; however, its robustness to out-of-distribution data is still challenging. This notebook describes our mdok approach in robust detection, based on fine-tuning smaller LLMs for text classification. It is applied to both subtasks of Voight-Kampff Generative AI Detection 2025, providing remarkable performance in binary detection as well as in multiclass (1st rank) classification of various cases of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01698v1">When LLMs Team Up: The Emergence of Collaborative Affective Computing</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 7 figures, and 3 tables
    </div>
    <details class="paper-abstract">
      Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v1">SALAD: Systematic Assessment of Machine Unlearing on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01687v1">StochasTok: Improving Fine-Grained Subword Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still often struggle with seemingly simple subword-level tasks like How many 'r's in 'strawberry'?. A key factor behind these failures is tokenization which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to 'see' their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs' downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok's simplicity allows seamless integration at any stage of the training pipeline; and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models. Code open-sourced at: https://github.com/anyasims/stochastok.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01646v1">ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 37 pages, 8 figures, 11 tables
    </div>
    <details class="paper-abstract">
      We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01631v1">Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance. To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01616v1">MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The emergence of multimodal LLM-based agents (MLAs) has transformed interaction paradigms by seamlessly integrating vision, language, action and dynamic environments, enabling unprecedented autonomous capabilities across GUI applications ranging from web automation to mobile systems. However, MLAs introduce critical trustworthiness challenges that extend far beyond traditional language models' limitations, as they can directly modify digital states and trigger irreversible real-world consequences. Existing benchmarks inadequately tackle these unique challenges posed by MLAs' actionable outputs, long-horizon uncertainty and multimodal attack vectors. In this paper, we introduce MLA-Trust, the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: truthfulness, controllability, safety and privacy. We utilize websites and mobile applications as realistic testbeds, designing 34 high-risk interactive tasks and curating rich evaluation datasets. Large-scale experiments involving 13 state-of-the-art agents reveal previously unexplored trustworthiness vulnerabilities unique to multimodal interactive scenarios. For instance, proprietary and open-source GUI-interacting MLAs pose more severe trustworthiness risks than static MLLMs, particularly in high-stakes domains; the transition from static MLLMs into interactive MLAs considerably compromises trustworthiness, enabling harmful content generation in multi-step interactions that standalone MLLMs would typically prevent; multi-step execution, while enhancing the adaptability of MLAs, involves latent nonlinear risk accumulation across successive interactions, circumventing existing safeguards and resulting in unpredictable derived risks. Moreover, we present an extensible toolbox to facilitate continuous evaluation of MLA trustworthiness across diverse interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v1">EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01484v1">LLM in the Loop: Creating the PARADEHATE Dataset for Hate Speech Detoxification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct PARADEHATE, a large-scale parallel dataset specifically for hatespeech detoxification. We release PARADEHATE as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on PARADEHATE, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01407v1">Comparing LLM-generated and human-authored news text using formal syntactic theory</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 15 figures, 13 tables; accepted to ACL-2025 main
    </div>
    <details class="paper-abstract">
      This study provides the first comprehensive comparison of New York Times-style text generated by six large language models against real, human-authored NYT writing. The comparison is based on a formal syntactic theory. We use Head-driven Phrase Structure Grammar (HPSG) to analyze the grammatical structure of the texts. We then investigate and illustrate the differences in the distributions of HPSG grammar types, revealing systematic distinctions between human and LLM-generated writing. These findings contribute to a deeper understanding of the syntactic behavior of LLMs as well as humans, within the NYT genre.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01386v1">ThinkEval: Practical Evaluation of Knowledge Preservation and Consistency in LLM Editing with Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Model editing has become an important tool for addressing privacy, bias, and misinformation in large language models (LLMs) by enabling updates to knowledge without the need for retraining from scratch. However, existing editing techniques often target isolated facts, ignoring ripple effects on related knowledge, allowing edited facts to remain deducible and compromising broader contextual integrity. For example, changing Harry Potter's school from Hogwarts to Ilvermorny requires reassigning his house from Gryffindor to a suitable alternative while preserving Gryffindor's relationship with Hogwarts. In this work, we present a new model-editing setting, deep editing, to show: (1) how editing techniques fail to handle connected facts, evaluating how original knowledge sneaks through unchanged causal links, and (2) their impact on broader contextual knowledge. We introduce ThinkEval, a framework to systematically evaluate model-editing techniques by building model-specific knowledge graphs to analyze pre- and post-edit effects on fact persistence and catastrophic forgetting. We present KnowGIC, a benchmark created with ThinkEval, consisting of sequentially linked queries to measure these effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. We find that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge. Our dataset is available at: https://anonymous.4open.science/r/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01374v1">Compiler Optimization via LLM Reasoning for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimization to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process, guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-aware transformations that reflect the current program state and accumulated performance feedback. Monte Carlo tree search (MCTS) incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01370v1">PointT2I: LLM-based text-to-image generation via keypoints</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generation model has made significant advancements, resulting in high-quality images aligned with an input prompt. However, despite T2I generation's ability to generate fine-grained images, it still faces challenges in accurately generating images when the input prompt contains complex concepts, especially human pose. In this paper, we propose PointT2I, a framework that effectively generates images that accurately correspond to the human pose described in the prompt by using a large language model (LLM). PointT2I consists of three components: Keypoint generation, Image generation, and Feedback system. The keypoint generation uses an LLM to directly generate keypoints corresponding to a human pose, solely based on the input prompt, without external references. Subsequently, the image generation produces images based on both the text prompt and the generated keypoints to accurately reflect the target pose. To refine the outputs of the preceding stages, we incorporate an LLM-based feedback system that assesses the semantic consistency between the generated contents and the given prompts. Our framework is the first approach to leveraging LLM for keypoints-guided image generation without any fine-tuning, producing accurate pose-aligned images based solely on textual prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01369v1">Incentivizing LLMs to Self-Verify Their Answers</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance during inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating its capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling. Our code is available at https://github.com/mansicer/self-verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01339v1">Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Machine unlearning offers a promising solution to privacy and safety concerns in large language models (LLMs) by selectively removing targeted knowledge while preserving utility. However, current methods are highly sensitive to downstream fine-tuning, which can quickly recover forgotten information-even from unrelated tasks. To address this, we introduce invariance into unlearning for the first time, inspired by invariant risk minimization (IRM). Building on this principle, we propose invariant LLM unlearning (ILU), a regularization-based framework that enhances robustness. Notably, ILU generalizes well to diverse fine-tuning tasks, even when trained using a single dataset. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP and MUSE benchmark, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01334v1">Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) decompose image classification into a process governed by interpretable, human-readable concepts. Recent advances in CBMs have used Large Language Models (LLMs) to generate candidate concepts. However, a critical question remains: What is the optimal number of concepts to use? Current concept banks suffer from redundancy or insufficient coverage. To address this issue, we introduce a dynamic, agent-based approach that adjusts the concept bank in response to environmental feedback, optimizing the number of concepts for sufficiency yet concise coverage. Moreover, we propose Conditional Concept Bottleneck Models (CoCoBMs) to overcome the limitations in traditional CBMs' concept scoring mechanisms. It enhances the accuracy of assessing each concept's contribution to classification tasks and feature an editable matrix that allows LLMs to correct concept scores that conflict with their internal knowledge. Our evaluations across 6 datasets show that our method not only improves classification accuracy by 6% but also enhances interpretability assessments by 30%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01290v1">TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating is built on the assumption that LLMs inherit ample knowledge, acquired during their extensive pretraining, enabling them to comprehend and discern quality differences in diverse TS data. We verify this assumption by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions via TSRater's inference on future TS samples. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01262v1">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Personalized AI assistants, a hallmark of the human-like capabilities of Large Language Models (LLMs), are a challenging application that intertwines multiple problems in LLM research. Despite the growing interest in the development of personalized assistants, the lack of an open-source conversational dataset tailored for personalization remains a significant obstacle for researchers in the field. To address this research gap, we introduce HiCUPID, a new benchmark to probe and unleash the potential of LLMs to deliver personalized responses. Alongside a conversational dataset, HiCUPID provides a Llama-3.2-based automated evaluation model whose assessment closely mirrors human preferences. We release our dataset, evaluation model, and code at https://github.com/12kimih/HiCUPID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01252v1">MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: https://github.com/Wayyuanyuan/MTCMB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01245v1">Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01237v1">Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: https://github.com/SungHo3268/KoGEM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00782v1">Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 21 pages, 8 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose \ourapproach, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that \ourapproach effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09577v3">FlashNorm: fast normalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper presents FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. FlashNorm also speeds up Layer Normalization and its recently proposed replacement Dynamic Tanh (DyT) arXiv:2503.10622. FlashNorm also reduces the number of parameter tensors by simply merging the normalization weights with the weights of the next linear layer. See https://github.com/OpenMachine-ai/transformer-tricks for code and more transformer tricks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12185v2">EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00657v2">LLM Safety Alignment is Divergence Estimation in Disguise</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      We present a theoretical framework showing that popular LLM alignment methods, including RLHF and its variants, can be understood as divergence estimators between aligned (safe or preferred) and unaligned (harmful or less preferred) distributions. This perspective explains the emergence of separation in the latent space between safe and harmful prompts after alignment. As an application of our general divergence framework, we propose KLDO, a novel KL divergence-based alignment method, and empirically validate its effectiveness. We further show that using compliance-refusal datasets, rather than standard preference-based datasets, leads to stronger separation and improved safety alignment. Finally, to quantify the separation effect, we propose a distance-based metric in the prompt representation space, which also acts as a statistically significant indicator for model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22877v2">Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Fact-checking is a potentially useful application of Large Language Models (LLMs) to combat the growing dissemination of disinformation. However, the performance of LLMs varies across geographic regions. In this paper, we evaluate the factual accuracy of open and private models across a diverse set of regions and scenarios. Using a dataset containing 600 fact-checked statements balanced across six global regions we examine three experimental setups of fact-checking a statement: (1) when just the statement is available, (2) when an LLM-based agent with Wikipedia access is utilized, and (3) as a best case scenario when a Retrieval-Augmented Generation (RAG) system provided with the official fact check is employed. Our findings reveal that regardless of the scenario and LLM used, including GPT-4, Claude Sonnet, and LLaMA, statements from the Global North perform substantially better than those from the Global South. Furthermore, this gap is broadened for the more realistic case of a Wikipedia agent-based system, highlighting that overly general knowledge bases have a limited ability to address region-specific nuances. These results underscore the urgent need for better dataset balancing and robust retrieval strategies to enhance LLM fact-checking capabilities, particularly in geographically diverse contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13975v2">Navigating Rifts in Human-LLM Grounding: Study and Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025, 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Language models excel at following instructions but often struggle with the collaborative aspects of conversation that humans naturally employ. This limitation in grounding -- the process by which conversation participants establish mutual understanding -- can lead to outcomes ranging from frustrated users to serious consequences in high-stakes scenarios. To systematically study grounding challenges in human-LLM interactions, we analyze logs from three human-assistant datasets: WildChat, MultiWOZ, and Bing Chat. We develop a taxonomy of grounding acts and build models to annotate and forecast grounding behavior. Our findings reveal significant differences in human-human and human-LLM grounding: LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans. Additionally, we find that early grounding failures predict later interaction breakdowns. Building on these insights, we introduce Rifts, a benchmark derived from publicly available LLM interaction data containing situations where LLMs fail to initiate grounding. We note that current frontier models perform poorly on Rifts, highlighting the need to reconsider how we train and prompt LLMs for human interaction. To this end, we develop a preliminary intervention aimed at mitigating grounding failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05873v2">MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL Findings 2025
    </div>
    <details class="paper-abstract">
      English-centric large language models (LLMs) often show strong multilingual capabilities. However, their multilingual performance remains unclear and is under-evaluated for many other languages. Most benchmarks for multilinguality focus on classic NLP tasks or cover a minimal number of languages. We introduce MEXA, a method for assessing the multilingual capabilities of pre-trained English-centric LLMs using parallel sentences, which are available for more languages than existing downstream tasks. MEXA leverages that English-centric LLMs use English as a pivot language in their intermediate layers. MEXA computes the alignment between English and non-English languages using parallel sentences to evaluate the transfer of language understanding from English to other languages. This alignment can be used to estimate model performance in different languages. We conduct controlled experiments using various parallel datasets (FLORES-200 and Bible), models (Llama family, Gemma family, Mistral, and OLMo), and established downstream tasks (Belebele, m-MMLU, and m-ARC). We explore different methods to compute embeddings in decoder-only models. Our results show that MEXA, in its default settings, achieves an average Pearson correlation of 0.90 between its predicted scores and actual task performance across languages. This suggests that MEXA is a reliable method for estimating the multilingual capabilities of English-centric LLMs, providing a clearer understanding of their multilingual potential and the inner workings of LLMs. Leaderboard: https://cis-lmu-mexa.hf.space, Code: https://github.com/cisnlp/MEXA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v2">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12728v3">Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Analogical reasoning is a unique ability of humans to address unfamiliar challenges by transferring strategies from relevant past experiences. One key finding in psychology is that compared with irrelevant past experiences, recalling relevant ones can help humans better handle new tasks. Coincidentally, the NLP community has also recently found that self-generating relevant examples in the context can help large language models (LLMs) better solve a given problem than hand-crafted prompts. However, it is yet not clear whether relevance is the key factor eliciting such capability, i.e., can LLMs benefit more from self-generated relevant examples than irrelevant ones? In this work, we systematically explore whether LLMs can truly perform analogical reasoning on a diverse set of reasoning tasks. With extensive experiments and analysis, we show that self-generated random examples can surprisingly achieve comparable or even better performance on certain tasks, e.g., 4% performance boost on GSM8K with random biological examples. We find that the accuracy of self-generated examples is the key factor and subsequently design two novel methods with improved performance and significantly reduced inference costs. Overall, we aim to advance a deeper understanding of LLM analogical reasoning and hope this work stimulates further research in the design of self-generated contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11191v2">Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.88% improvement in the aggregate score, while reasoning distillation leads to a 10% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18458v3">A Survey of LLM $\times$ DATA</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Please refer to the paper list at: https://github.com/weAIDB/awesome-data-llm
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20238v2">FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL2025 Main
    </div>
    <details class="paper-abstract">
      Many challenging reasoning tasks require not just rapid, intuitive responses, but a more deliberate, multi-step approach. Recent progress in large language models (LLMs) highlights an important shift from the "System 1" way of quick reactions to the "System 2" style of reflection-and-correction problem solving. However, current benchmarks heavily rely on the final-answer accuracy, leaving much of a model's intermediate reasoning steps unexamined. This fails to assess the model's ability to reflect and rectify mistakes within the reasoning process. To bridge this gap, we introduce FINEREASON, a logic-puzzle benchmark for fine-grained evaluation of LLMs' reasoning capabilities. Each puzzle can be decomposed into atomic steps, making it ideal for rigorous validation of intermediate correctness. Building on this, we introduce two tasks: state checking, and state transition, for a comprehensive evaluation of how models assess the current situation and plan the next move. To support broader research, we also provide a puzzle training set aimed at enhancing performance on general mathematical tasks. We show that models trained on our state checking and transition data demonstrate gains in math reasoning by up to 5.1% on GSM8K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10424v3">LLM-as-an-Interviewer: Beyond Static Testing Through Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      We introduce LLM-as-an-Interviewer, a novel paradigm for evaluating large language models (LLMs). This approach leverages multi-turn interactions where the LLM interviewer actively provides feedback on responses and poses follow-up questions to the evaluated LLM. At the start of the interview, the LLM interviewer dynamically modifies datasets to generate initial questions, mitigating data contamination. We apply the LLM-as-an-Interviewer framework to evaluate six models on the MATH and DepthQA tasks. Our results show that the framework effectively provides insights into LLM performance, including the quality of initial responses, adaptability to feedback, and ability to address follow-up queries like clarification or additional knowledge requests. The framework also addresses key limitations of conventional methods like LLM-as-a-Judge, including verbosity bias and inconsistency across runs. Finally, we propose the Interview Report, which aggregates insights from the interview process, providing examples and a comprehensive analysis of the LLM's strengths and weaknesses. This report offers a detailed snapshot of the model's real-world applicability. The code for our framework is publicly available at https://github.com/interview-eval/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19433v2">Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ICML2025 as Poster
    </div>
    <details class="paper-abstract">
      Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks only focus on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities - workflow, tool use/function call, long-context understanding and real-world application. We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ) and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%-3% drop) but degrades real-world application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios. The code can be found in https://github.com/pprp/ACBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16978v2">HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025 Findings. Code available at https://github.com/RutaTang/HyGenar
    </div>
    <details class="paper-abstract">
      Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02450v2">Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 2025 ACL Findings
    </div>
    <details class="paper-abstract">
      Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at https://github.com/SnowCharmQ/DPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18530v2">IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13602v2">GAMEBoT: Transparent Assessment of LLM Reasoning in Games</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 9 pages, ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: https://visual-ai.github.io/gamebot
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v4">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11196v2">How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at https://github.com/zjunlp/DynamicKnowledgeCircuits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11674v2">LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Time series forecasting using LLMs
    </div>
    <details class="paper-abstract">
      Time series forecasting remains a challenging task, particularly in the context of complex multiscale temporal patterns. This study presents LLM-Mixer, a framework that improves forecasting accuracy through the combination of multiscale time-series decomposition with pre-trained LLMs (Large Language Models). LLM-Mixer captures both short-term fluctuations and long-term trends by decomposing the data into multiple temporal resolutions and processing them with a frozen LLM, guided by a textual prompt specifically designed for time-series data. Extensive experiments conducted on multivariate and univariate datasets demonstrate that LLM-Mixer achieves competitive performance, outperforming recent state-of-the-art models across various forecasting horizons. This work highlights the potential of combining multiscale analysis and LLMs for effective and scalable time-series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v4">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Camera
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v3">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup. The code is available at https://github.com/gszfwsb/Data-Whisperer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10084v2">Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Despite the remarkable successes of large language models (LLMs), the underlying Transformer architecture has inherent limitations in handling complex reasoning tasks. Chain-of-thought (CoT) prompting has emerged as a practical workaround, but most CoT-based methods rely on a single, generic prompt such as "think step by step", with no task-specific adaptation. These approaches expect the model to discover an effective reasoning path on its own, forcing it to search through a vast prompt space. In contrast, several studies have explored task-specific prompt designs to boost performance. However, these designs are typically developed through trial and error, lacking theoretical grounding. As a result, prompt engineering remains largely ad hoc and unguided. In this paper, we provide a theoretical framework that explains why some prompts succeed while others fail. We show that prompts function as selectors, extracting task-relevant information from the model's full hidden state during CoT reasoning. Each prompt defines a unique trajectory through the answer space, and the choice of trajectory is crucial for task performance and future navigation within the space. We analyze the complexity of finding optimal prompts and characterize the size of the prompt space for a given task. Our theory reveals principles behind effective prompt design and shows that naive CoT-using self-guided prompts like "think step by step"-can severely hinder performance. Through experiments, we show that optimal prompt search can lead to more than a 50% improvement on reasoning tasks, providing a theoretical foundation for prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11368v2">LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus and code for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03219v2">Content Moderation by LLM: From Accuracy to Legitimacy</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      One trending application of LLM (large language model) is to use it for content moderation in online platforms. Most current studies on this application have focused on the metric of accuracy -- the extent to which LLMs make correct decisions about content. This article argues that accuracy is insufficient and misleading because it fails to grasp the distinction between easy cases and hard cases, as well as the inevitable trade-offs in achieving higher accuracy. Closer examination reveals that content moderation is a constitutive part of platform governance, the key of which is to gain and enhance legitimacy. Instead of making moderation decisions correct, the chief goal of LLMs is to make them legitimate. In this regard, this article proposes a paradigm shift from the single benchmark of accuracy towards a legitimacy-based framework for evaluating the performance of LLM moderators. The framework suggests that for easy cases, the key is to ensure accuracy, speed, and transparency, while for hard cases, what matters is reasoned justification and user participation. Examined under this framework, LLMs' real potential in moderation is not accuracy improvement. Rather, LLMs can better contribute in four other aspects: to conduct screening of hard cases from easy cases, to provide quality explanations for moderation decisions, to assist human reviewers in getting more contextual information, and to facilitate user participation in a more interactive way. To realize these contributions, this article proposes a workflow for incorporating LLMs into the content moderation system. Using normative theories from law and social sciences to critically assess the new technological application, this article seeks to redefine LLMs' role in content moderation and redirect relevant research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v2">Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Preprint; 27 pages, 12 figures, 10 tables; Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor environments and under open-vocabulary object descriptions. We observe that most episodes in widely used benchmarks such as HM3D and MP3D involve multi-floor buildings, with many requiring explicit floor transitions. However, existing methods are often limited to single-floor settings or predefined object categories. To address these limitations, we tackle two key challenges: (1) efficient cross-level planning and (2) zero-shot object-goal navigation (ZS-OGN), where agents must interpret novel object descriptions without prior exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine Frontier Reasoning module leveraging Large Language Models (LLMs) for context-aware exploration, without requiring additional training on new object semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor navigation. We further validate its practicality through real-world deployment on a quadruped robot, achieving successful object exploration across unseen floors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18792v2">REALM: A Dataset of Real-World LLM Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14127v2">Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing, where LLMs construct and explain answers, better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA, robustness, biases, and unfaithful explanations, showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01195v1">CoBRA: Quantifying Strategic Language Use and LLM Pragmatics</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Language is often used strategically, particularly in high-stakes, adversarial settings, yet most work on pragmatics and LLMs centers on cooperativity. This leaves a gap in systematic understanding of non-cooperative discourse. To address this, we introduce CoBRA (Cooperation-Breach Response Assessment), along with three interpretable metrics -- Benefit at Turn (BaT), Penalty at Turn (PaT), and Normalized Relative Benefit at Turn (NRBaT) -- to quantify the perceived strategic effects of discourse moves. We also present CHARM, an annotated dataset of real courtroom cross-examinations, to demonstrate the framework's effectiveness. Using these tools, we evaluate a range of LLMs and show that LLMs generally exhibit limited pragmatic understanding of strategic language. While model size shows an increase in performance on our metrics, reasoning ability does not help and largely hurts, introducing overcomplication and internal confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01190v1">Culturally-Grounded Chain-of-Thought (CG-CoT):Enhancing LLM Performance on Culturally-Specific Tasks in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with culturally-specific reasoning tasks, particularly in low-resource languages, hindering their global applicability. Addressing this gap is crucial for equitable AI deployment. We introduce Culturally-Grounded Chain-of-Thought (CG-CoT), a novel prompting strategy that combines dense vector retrieval of cultural context with explicit reasoning sequences. Our extensive experiments on Yoruba proverb interpretation demonstrate that CG-CoT provides significantly higher culturally-aligned accuracy and depth than traditional prompting methods, validated through both automated metrics and LLM-based evaluations. Notably, we uncover stark disparities between token-level translation metrics like BLEU and human-judged cultural relevance, suggesting a rethinking of evaluation approaches for low-resource NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01140v1">ReTern: Exploiting Natural Redundancy and Sign Transformations for Enhanced Fault Tolerance in Compute-in-Memory based Ternary LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Ternary large language models (LLMs), which utilize ternary precision weights and 8-bit activations, have demonstrated competitive performance while significantly reducing the high computational and memory requirements of full-precision LLMs. The energy efficiency and performance of Ternary LLMs can be further improved by deploying them on ternary computing-in-memory (TCiM) accelerators, thereby alleviating the von-Neumann bottleneck. However, TCiM accelerators are prone to memory stuck-at faults (SAFs) leading to degradation in the model accuracy. This is particularly severe for LLMs due to their low weight sparsity. To boost the SAF tolerance of TCiM accelerators, we propose ReTern that is based on (i) fault-aware sign transformations (FAST) and (ii) TCiM bit-cell reprogramming exploiting their natural redundancy. The key idea is to utilize FAST to minimize computations errors due to SAFs in +1/-1 weights, while the natural bit-cell redundancy is exploited to target SAFs in 0 weights (zero-fix). Our experiments on BitNet b1.58 700M and 3B ternary LLMs show that our technique furnishes significant fault tolerance, notably 35% reduction in perplexity on the Wikitext dataset in the presence of faults. These benefits come at the cost of < 3%, < 7%, and < 1% energy, latency and area overheads respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01116v1">ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01114v1">Reconsidering LLM Uncertainty Estimation Methods in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: https://github.com/duygunuryldz/uncertainty_in_the_wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01104v1">Contextual Candor: Enhancing LLM Trustworthiness Through Hierarchical Unanswerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The pervasive deployment of large language models (LLMs) in conversational AI systems has revolutionized information access, yet their propensity for generating factually unsupported or hallucinated responses remains a critical impediment to trustworthiness and widespread adoption. This paper introduces Reinforced Unanswerability Learning (RUL), a novel hybrid training paradigm designed to imbue LLMs with the intrinsic capability to accurately detect unanswerable questions and generate reliably appropriate responses. Unlike conventional approaches that rely on external classifiers or simple prompting, RUL integrates a discriminative unanswerability prediction head with the LLM's generative core, guided by a multi-stage learning strategy. This includes supervised fine-tuning on a novel, richly annotated dataset, Enhanced-CAsT-Answerability (ECA), which features hierarchical answerability labels and ground-truth refusal responses. Crucially, RUL incorporates a subsequent reinforcement learning with human feedback (RLHF) phase to refine the nuance, helpfulness, and informativeness of refusal responses. Extensive experiments demonstrate RUL's superior performance, achieving significantly higher accuracy in unanswerability detection across sentence, paragraph, and ranking levels, and substantially increasing the generation of appropriate refusals for unanswerable queries, alongside strong performance on answerable questions. Human evaluations further corroborate RUL's effectiveness, highlighting a marked improvement in perceived helpfulness and trustworthiness, ultimately paving the way for more reliable and user-centric conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01089v1">Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: https://github.com/metehanoguzz/LLMs-Indexicals-English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01063v1">AI4Contracts: LLM & RAG-Powered Encoding of Financial Derivative Contracts</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 8 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) are reshaping how AI systems extract and organize information from unstructured text. A key challenge is designing AI methods that can incrementally extract, structure, and validate information while preserving hierarchical and contextual relationships. We introduce CDMizer, a template-driven, LLM, and RAG-based framework for structured text transformation. By leveraging depth-based retrieval and hierarchical generation, CDMizer ensures a controlled, modular process that aligns generated outputs with predefined schema. Its template-driven approach guarantees syntactic correctness, schema adherence, and improved scalability, addressing key limitations of direct generation methods. Additionally, we propose an LLM-powered evaluation framework to assess the completeness and accuracy of structured representations. Demonstrated in the transformation of Over-the-Counter (OTC) financial derivative contracts into the Common Domain Model (CDM), CDMizer establishes a scalable foundation for AI-driven document understanding, structured synthesis, and automated validation in broader contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v1">MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds. The code and dataset will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01055v1">Simple Prompt Injection Attacks Can Leak Personal Data Observed by LLM Agents During Task Execution</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 25 pages, 18 figures, NeurIPS formatting style
    </div>
    <details class="paper-abstract">
      Previous benchmarks on prompt injection in large language models (LLMs) have primarily focused on generic tasks and attacks, offering limited insights into more complex threats like data exfiltration. This paper examines how prompt injection can cause tool-calling agents to leak personal data observed during task execution. Using a fictitious banking agent, we develop data flow-based attacks and integrate them into AgentDojo, a recent benchmark for agentic security. To enhance its scope, we also create a richer synthetic dataset of human-AI banking conversations. In 16 user tasks from AgentDojo, LLMs show a 15-50 percentage point drop in utility under attack, with average attack success rates (ASR) around 20 percent; some defenses reduce ASR to zero. Most LLMs, even when successfully tricked by the attack, avoid leaking highly sensitive data like passwords, likely due to safety alignments, but they remain vulnerable to disclosing other personal data. The likelihood of password leakage increases when a password is requested along with one or two additional personal details. In an extended evaluation across 48 tasks, the average ASR is around 15 percent, with no built-in AgentDojo defense fully preventing leakage. Tasks involving data extraction or authorization workflows, which closely resemble the structure of exfiltration attacks, exhibit the highest ASRs, highlighting the interaction between task type, agent performance, and defense efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02058v1">Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01049v1">Taming LLMs by Scaling Learning Rates with Gradient Grouping</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Preprint version of "Taming LLMs with Gradient Grouping" (ACL'2025). The code will be available at https://github.com/ScalingOpt/SGG
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) poses challenges due to their massive scale and heterogeneous architectures. While adaptive optimizers like AdamW help address gradient variations, they still struggle with efficient and effective parameter-wise learning rate estimation, resulting in training instability, slow convergence, and poor compatibility with parameter-efficient fine-tuning (PEFT) techniques. This work introduces Scaling with Gradient Grouping (SGG), an optimizer wrapper that improves adaptive learning rate estimation by dynamic grouping and group-specific scaling. SGG first groups gradient statistics in each layer into clusters and then applies cluster-specific scaling to calibrate learning rates for each parameter, thus imposing collective group-wise constraints while maintaining precise per-parameter adaptation. Experiments on diverse (M)LLM benchmarks show that SGG integrates seamlessly with existing optimizers, and offers consistent gains and faster convergence over baselines, with various model sizes. Its stability across varying batch sizes and learning rates establishes SGG as a robust choice for LLM optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01048v1">IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional performance across a wide range of natural language tasks. However, selecting the optimal LLM to respond to a user query often necessitates a delicate balance between performance and cost. While powerful models deliver better results, they come at a high cost, whereas smaller models are more cost-effective but less capable. To address this trade-off, we propose IRT-Router, a multi-LLM routing framework that efficiently routes user queries to the most suitable LLM. Inspired by Item Response Theory (IRT), a psychological measurement methodology, IRT-Router explicitly models the relationship between LLM capabilities and user query attributes. This not only enables accurate prediction of response performance but also provides interpretable insights, such as LLM abilities and query difficulty. Additionally, we design an online query warm-up technique based on semantic similarity, further enhancing the online generalization capability of IRT-Router. Extensive experiments on 20 LLMs and 12 datasets demonstrate that IRT-Router outperforms most baseline methods in terms of effectiveness and interpretability. Its superior performance in cold-start scenarios further confirms the reliability and practicality of IRT-Router in real-world applications. Code is available at https://github.com/Mercidaiha/IRT-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00998v1">LoRA-BAM: Input Filtering for Fine-tuned LLMs via Boxed Abstraction Monitors over LoRA Layers</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance on domain-specific tasks but can lead to overfitting, making them unreliable on out-of-distribution (OoD) queries. We propose LoRA-BAM - a method that adds OoD detection monitors to the LoRA layer using boxed abstraction to filter questions beyond the model's competence. Feature vectors from the fine-tuning data are extracted via the LLM and clustered. Clusters are enclosed in boxes; a question is flagged as OoD if its feature vector falls outside all boxes. To improve interpretability and robustness, we introduce a regularization loss during fine-tuning that encourages paraphrased questions to stay close in the feature space, and the enlargement of the decision boundary is based on the feature variance within a cluster. Our method complements existing defenses by providing lightweight and interpretable OoD detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00985v1">Do LLMs Understand Why We Write Diaries? A Method for Purpose Extraction and Clustering</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted for CompLing-2025 conference
    </div>
    <details class="paper-abstract">
      Diary analysis presents challenges, particularly in extracting meaningful information from large corpora, where traditional methods often fail to deliver satisfactory results. This study introduces a novel method based on Large Language Models (LLMs) to identify and cluster the various purposes of diary writing. By "purposes," we refer to the intentions behind diary writing, such as documenting life events, self-reflection, or practicing language skills. Our approach is applied to Soviet-era diaries (1922-1929) from the Prozhito digital archive, a rich collection of personal narratives. We evaluate different proprietary and open-source LLMs, finding that GPT-4o and o1-mini achieve the best performance, while a template-based baseline is significantly less effective. Additionally, we analyze the retrieved purposes based on gender, age of the authors, and the year of writing. Furthermore, we examine the types of errors made by the models, providing a deeper understanding of their limitations and potential areas for improvement in future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00888v1">An Integrated Platform for LEED Certification Automation Using Computer Vision and LLM-RAG</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The Leadership in Energy and Environmental Design (LEED) certification process is characterized by labor-intensive requirements for data handling, simulation, and documentation. This paper presents an automated platform designed to streamline key aspects of LEED certification. The platform integrates a PySide6-based user interface, a review Manager for process orchestration, and multiple analysis engines for credit compliance, energy modeling via EnergyPlus, and location-based evaluation. Key components include an OpenCV-based preprocessing pipeline for document analysis and a report generation module powered by the Gemma3 large language model with a retrieval-augmented generation framework. Implementation techniques - including computer vision for document analysis, structured LLM prompt design, and RAG-based report generation - are detailed. Initial results from pilot project deployment show improvements in efficiency and accuracy compared to traditional manual workflows, achieving 82% automation coverage and up to 70% reduction in documentation time. The platform demonstrates practical scalability for green building certification automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00845v1">Generalizable LLM Learning of Graph Synthetic Data with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 9 pages, 3 figures, 3 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
    </div>
    <details class="paper-abstract">
      Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph synthetic data with reinforcement learning. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that RL would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting. We employ RL algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our RL recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9\% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards, mixing synthetic and real-world task data yields potential gains, while compositionality and explainable intermediate steps remains a critical challenge even after RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00844v1">LLM Cannot Discover Causality, and Should Be Restricted to Non-Decisional Support in Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      This paper critically re-evaluates LLMs' role in causal discovery and argues against their direct involvement in determining causal relationships. We demonstrate that LLMs' autoregressive, correlation-driven modeling inherently lacks the theoretical grounding for causal reasoning and introduces unreliability when used as priors in causal discovery algorithms. Through empirical studies, we expose the limitations of existing LLM-based methods and reveal that deliberate prompt engineering (e.g., injecting ground-truth knowledge) could overstate their performance, helping to explain the consistently favorable results reported in much of the current literature. Based on these findings, we strictly confined LLMs' role to a non-decisional auxiliary capacity: LLMs should not participate in determining the existence or directionality of causal relationships, but can assist the search process for causal graphs (e.g., LLM-based heuristic search). Experiments across various settings confirm that, by strictly isolating LLMs from causal decision-making, LLM-guided heuristic search can accelerate the convergence and outperform both traditional and LLM-based methods in causal structure learning. We conclude with a call for the community to shift focus from naively applying LLMs to developing specialized models and training method that respect the core principles of causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00823v1">Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 19 pages, 16 figures; accepted to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on extensive datasets that encapsulate substantial world knowledge. However, their outputs often include confidently stated inaccuracies. Earlier works suggest that LLMs encode truthfulness as a distinct linear feature, termed the "truth direction", which can classify truthfulness reliably. We address several open questions about the truth direction: (i) whether LLMs universally exhibit consistent truth directions; (ii) whether sophisticated probing techniques are necessary to identify truth directions; and (iii) how the truth direction generalizes across diverse contexts. Our findings reveal that not all LLMs exhibit consistent truth directions, with stronger representations observed in more capable models, particularly in the context of logical negation. Additionally, we demonstrate that truthfulness probes trained on declarative atomic statements can generalize effectively to logical transformations, question-answering tasks, in-context learning, and external knowledge sources. Finally, we explore the practical application of truthfulness probes in selective question-answering, illustrating their potential to improve user trust in LLM outputs. These results advance our understanding of truth directions and provide new insights into the internal representations of LLM beliefs. Our code is public at https://github.com/colored-dye/truthfulness_probe_generalization
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00807v1">Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of large language models (LLMs) have significantly advanced their performance by enabling in-depth understanding of diverse tasks. With growing interest in applying LLMs to the time series domain, this has proven nontrivial, as evidenced by the limited efficacy of straightforwardly adapting text-domain reasoning techniques. Although recent work has shown promise in several time series tasks, further leveraging advancements in LLM reasoning remains under-explored for time series classification (TSC) tasks, despite their prevalence and significance in many real-world applications. In this paper, we propose ReasonTSC, a novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC. Rather than straightforwardly applying existing reasoning techniques or relying solely on LLMs' built-in reasoning capabilities, ReasonTSC first steers the model to think over the essential characteristics of time series data. Next, it integrates predictions and confidence scores from plug-in classifiers, e.g., domain-specific time series models, as in-context examples. Finally, ReasonTSC guides the LLM through a structured reasoning process: it evaluates the initial assessment, backtracks to consider alternative hypotheses, and compares their merits before arriving at a final classification. Extensive experiments and systematic ablation studies demonstrate that ReasonTSC consistently outperforms both existing time series reasoning baselines and plug-in models, and is even capable of identifying and correcting plug-in models' false predictions.
    </details>
</div>
