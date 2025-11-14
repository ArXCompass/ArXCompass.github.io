# llm - 2025_09

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
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11967v1">MillStone: How Open-Minded Are LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 19 pages, 7 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11921v1">Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v2">Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v2">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11864v1">NeuroStrike: Neuron-Level Attacks on Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability. This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11803v1">From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: https://github.com/lielsheri/PatientSignal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v5">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v5">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20258v2">LLM as a Broken Telephone: Iterative Generation Distorts Information</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to ACL 2025, Main Conference
    </div>
    <details class="paper-abstract">
      As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08575v2">SQLGovernor: An LLM-powered SQL Toolkit for Real World Application</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      SQL queries in real world analytical environments, whether written by humans or generated automatically often suffer from syntax errors, inefficiency, or semantic misalignment, especially in complex OLAP scenarios. To address these challenges, we propose SQLGovernor, an LLM powered SQL toolkit that unifies multiple functionalities, including syntax correction, query rewriting, query modification, and consistency verification within a structured framework enhanced by knowledge management. SQLGovernor introduces a fragment wise processing strategy to enable fine grained rewriting and localized error correction, significantly reducing the cognitive load on the LLM. It further incorporates a hybrid self learning mechanism guided by expert feedback, allowing the system to continuously improve through DBMS output analysis and rule validation. Experiments on benchmarks such as BIRD and BIRD CRITIC, as well as industrial datasets, show that SQLGovernor consistently boosts the performance of base models by up to 10%, while minimizing reliance on manual expertise. Deployed in production environments, SQLGovernor demonstrates strong practical utility and effective performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20742v2">'Hello, World!': Making GNNs Talk with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Published as a conference paper at EMNLP 2025 Findings. Code and datasets are in https://github.com/kswoo97/GLN-Code
    </div>
    <details class="paper-abstract">
      While graph neural networks (GNNs) have shown remarkable performance across diverse graph-related tasks, their high-dimensional hidden representations render them black boxes. In this work, we propose Graph Lingual Network (GLN), a GNN built on large language models (LLMs), with hidden representations in the form of human-readable text. Through careful prompt design, GLN incorporates not only the message passing module of GNNs but also advanced GNN techniques, including graph attention and initial residual connection. The comprehensibility of GLN's hidden representations enables an intuitive analysis of how node representations change (1) across layers and (2) under advanced GNN techniques, shedding light on the inner workings of GNNs. Furthermore, we demonstrate that GLN achieves strong zero-shot performance on node classification and link prediction, outperforming existing LLM-based baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12172v3">Navigating the Labyrinth: Evaluating LLMs' Ability to Reason About Search Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved impressive performance in math and reasoning benchmarks. However, they often struggle with logic problems and puzzles that are relatively easy for humans. To further investigate this, we introduce a new benchmark, SearchBench, which contains 11 unique search problems inspired by intuitive puzzles. Each SearchBench problem type is equipped with automated pipelines to generate an arbitrary number of instances and analyze the feasibility, correctness, and optimality of LLM-generated solutions. We show that using step-by-step, language-only reasoning, even the most advanced LLMs fail to solve SearchBench; for example, OpenAI's frontier models GPT-4 and o1-preview solve only 1.4% and 18.6% of problems, respectively. The reason is that SearchBench problems require considering multiple pathways and performing backtracking, posing a significant challenge to auto-regressive models. Interestingly, performance is significantly boosted when we prompt models to generate a complete A* search algorithm - a comparatively more cognitively difficult task. This approach effectively offloads the iterative search and backtracking process from the models, which they struggle with in text. This in-context learning baseline is further enhanced via a Multi-Stage-Multi-Try (MSMT) inference method, increasing GPT-4's rate of correct solutions to over 57%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11617v1">AssemMate: Graph-Based LLM for Robotic Assembly Assistance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: https://github.com/cristina304/AssemMate.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03054v2">Binary Quantization For LLMs Through Dynamic Grouping</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 An error was identified in the quantization bit width; it is not binary
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of Natural Language Processing (NLP) tasks, but require substantial memory and computational resources. Binary quantization, which compresses model weights from 16-bit Brain Float to 1-bit representations in {-1, 1}, offers significant reductions in storage and inference costs. However, such aggressive quantization often leads to notable performance degradation compared to more conservative 4-bit quantization methods. In this research, we propose a novel optimization objective tailored for binary quantization, along with three algorithms designed to realize it effectively. Our method enhances blocked quantization by dynamically identifying optimal unstructured sub-matrices through adaptive grouping strategies. Experimental results demonstrate that our approach achieves an average bit length of just 1.007 bits, while maintaining high model quality. Specifically, our quantized LLaMA 3.2 3B model attains a perplexity of 8.23, remarkably close to the original 7.81, and surpasses previous SOTA BiLLM with a perplexity of only 123.90. Furthermore, our method is competitive with SOTA 4-bit approaches such as GPTQ in both performance and efficiency. The compression process is highly efficient, requiring only 14 seconds to quantize the full LLaMA 3.2 3B weights on a single CPU core, with the entire process completing in under 100 minutes and exhibiting embarrassingly parallel properties. Code - https://github.com/johnnyzheng0636/WGM_bi_quan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11569v1">D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v2">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16347v2">Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00063v2">MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11362v1">PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Understanding human behavior traits is central to applications in human-computer interaction, computational social science, and personalized AI systems. Such understanding often requires integrating multiple modalities to capture nuanced patterns and relationships. However, existing resources rarely provide datasets that combine behavioral descriptors with complementary modalities such as facial attributes and biographical information. To address this gap, we present PersonaX, a curated collection of multimodal datasets designed to enable comprehensive analysis of public traits across modalities. PersonaX consists of (1) CelebPersona, featuring 9444 public figures from diverse occupations, and (2) AthlePersona, covering 4181 professional athletes across 7 major sports leagues. Each dataset includes behavioral trait assessments inferred by three high-performing large language models, alongside facial imagery and structured biographical features. We analyze PersonaX at two complementary levels. First, we abstract high-level trait scores from text descriptions and apply five statistical independence tests to examine their relationships with other modalities. Second, we introduce a novel causal representation learning (CRL) framework tailored to multimodal and multi-measurement data, providing theoretical identifiability guarantees. Experiments on both synthetic and real-world data demonstrate the effectiveness of our approach. By unifying structured and unstructured analysis, PersonaX establishes a foundation for studying LLM-inferred behavioral traits in conjunction with visual and biographical attributes, advancing multimodal trait analysis and causal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11353v1">Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in information systems, including being used as second-stage rerankers in information retrieval pipelines, yet their susceptibility to recency bias has received little attention. We investigate whether LLMs implicitly favour newer documents by prepending artificial publication dates to passages in the TREC Deep Learning passage retrieval collections in 2021 (DL21) and 2022 (DL22). Across seven models, GPT-3.5-turbo, GPT-4o, GPT-4, LLaMA-3 8B/70B, and Qwen-2.5 7B/72B, "fresh" passages are consistently promoted, shifting the Top-10's mean publication year forward by up to 4.78 years and moving individual items by as many as 95 ranks in our listwise reranking experiments. Although larger models attenuate the effect, none eliminate it. We also observe that the preference of LLMs between two passages with an identical relevance level can be reversed by up to 25% on average after date injection in our pairwise preference experiments. These findings provide quantitative evidence of a pervasive recency bias in LLMs and highlight the importance of effective bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17573v2">Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Accepted in EMNLP main proceedings; Updated citations
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11451v2">From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable solutions with reduced costs and enhanced data privacy. This paper explores the role of personas in the creation of ESC by LLMs. Our research utilizes established psychological frameworks to measure and infuse persona traits into LLMs, which then generate dialogues in the emotional support scenario. We conduct extensive evaluations to understand the stability of persona traits in dialogues, examining shifts in traits post-generation and their impact on dialogue quality and strategy distribution. Experimental results reveal several notable findings: 1) LLMs can infer core persona traits, 2) subtle shifts in emotionality and extraversion occur, influencing the dialogue dynamics, and 3) the application of persona traits modifies the distribution of emotional support strategies, enhancing the relevance and empathetic quality of the responses. These findings highlight the potential of persona-driven LLMs in crafting more personalized, empathetic, and effective emotional support dialogues, which has significant implications for the future design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20022v2">Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11311v1">Prompts to Proxies: Emulating Human Preferences via a Compact LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Preprint of work originally submitted to AAAI 2026. Under revision for resubmission to a machine learning venue
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promise in emulating human-like responses across a wide range of tasks. In this paper, we propose a novel alignment framework that treats LLMs as agent proxies for human survey respondents, affording a cost-effective and steerable solution to two pressing challenges in the social sciences: the rising cost of survey deployment and the growing demographic imbalance in survey response data. Drawing inspiration from the theory of revealed preference, we formulate alignment as a two-stage problem: constructing diverse agent personas called endowments that simulate plausible respondent profiles, and selecting a representative subset to approximate a ground-truth population based on observed data. To implement the paradigm, we introduce P2P, a system that steers LLM agents toward representative behavioral patterns using structured prompt engineering, entropy-based sampling, and regression-based selection. Unlike personalization-heavy approaches, our alignment approach is demographic-agnostic and relies only on aggregate survey results, offering better generalizability and parsimony. Beyond improving data efficiency in social science research, our framework offers a testbed for studying the operationalization of pluralistic alignment. We demonstrate the efficacy of our approach on real-world opinion survey datasets, showing that our aligned agent populations can reproduce aggregate response patterns with high fidelity and exhibit substantial response diversity, even without demographic conditioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15214v2">Think Small, Plan Smart: Minimalist Symbolic Abstraction and Heuristic Subspace Search for LLM-Guided Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Reliable task planning is pivotal for achieving long-horizon autonomy in real-world robotic systems. Large language models (LLMs) offer a promising interface for translating complex and ambiguous natural language instructions into actionable plans. However, their probabilistic and opaque nature often leads to logically inconsistent or infeasible outputs. To address these limitations, recent frameworks combine LLMs with symbolic planners by first generating action models (Planning Domain Definition Language) and then applying heuristic search. Although promising, such systems still suffer from representation redundancy and exponential search complexity, often resulting in inefficient or overly long plans. To improve planning efficiency and effectiveness, we propose PLAHX (Planning from Language using Abstraction and Heuristic eXploration), a two-stage LLM-symbolic planning framework that integrates abstract symbolic representations with meta-heuristic subspace search in a parallel and iterative fashion. Rather than relying on verbose LLM-generated domain models, we introduce a minimalist symbolic abstraction pipeline that preserves semantic fidelity while eliminating redundancy. Our approach redefines LLM-symbolic planning not by making LLMs smarter, but by reducing the symbolic search space adaptively. Empirical results across four challenging domains, including block stacking and robotic mobile grasping, show that our approach improves the success rate by 21.47% on average, while reducing token consumption by 13% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18998v2">Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/knowledge-verse-ai/LLM-Memorization_SK_Eval-.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11177v1">Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11889v2">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Recent studies have explored integrating large language models (LLMs) into recommendation systems but face several challenges, including training-induced bias and bottlenecks from serialized architecture. To effectively address these issues, we propose a Query-toRecommendation, a parallel recommendation framework that decouples LLMs from candidate pre-selection and instead enables direct retrieval over the entire item pool. Our framework connects LLMs and recommendation models in a parallel manner, allowing each component to independently utilize its strengths without interfering with the other. In this framework, LLMs are utilized to generate feature-enriched item descriptions and personalized user queries, allowing for capturing diverse preferences and enabling rich semantic matching in a zero-shot manner. To effectively combine the complementary strengths of LLM and collaborative signals, we introduce an adaptive reranking strategy. Extensive experiments demonstrate an improvement in performance up to 57%, while also improving the novelty and diversity of recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12805v2">Assessing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Corrected a typo in the metadata title only ("Assesing"->"Assessing"). No changes were made to the PDF or source files
    </div>
    <details class="paper-abstract">
      This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11155v1">AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09112v2">Character-Level Perturbations Disrupt LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries. To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04827v2">VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving systems increasingly support interactive applications, like real-time chat assistants, code generation tools, and agentic workflows. However, the soaring energy cost of LLM inference presents a growing challenge for sustainable and cost-effective deployment. This paper introduces VoltanaLLM, a system for SLO-aware, energy-efficient LLM serving, built from a control theory perspective. VoltanaLLM co-designs frequency scaling and request routing in emerging prefill/decode disaggregated architectures, leveraging their decoupled execution to enable fine-grained phase-specific control. It consists of a feedback-driven frequency controller that dynamically adapts GPU frequency for prefill and decode phases, and a state-space router that explores routing decisions across frequency-scaled instances to minimize energy under latency constraints. We implement VoltanaLLM in SGLang and evaluate its performance over multiple state-of-the-art LLMs and real-world datasets. The results demonstrate that VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rate, paving the way for sustainable and intelligent LLM serving. Code of VoltanaLLM is open-sourced on GitHub: https://github.com/Supercomputing-System-AI-Lab/VoltanaLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11141v1">When Smiley Turns Hostile: Interpreting How Emojis Trigger LLMs' Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Emojis are globally used non-verbal cues in digital communication, and extensive research has examined how large language models (LLMs) understand and utilize emojis across contexts. While usually associated with friendliness or playfulness, it is observed that emojis may trigger toxic content generation in LLMs. Motivated by such a observation, we aim to investigate: (1) whether emojis can clearly enhance the toxicity generation in LLMs and (2) how to interpret this phenomenon. We begin with a comprehensive exploration of emoji-triggered LLM toxicity generation by automating the construction of prompts with emojis to subtly express toxic intent. Experiments across 5 mainstream languages on 7 famous LLMs along with jailbreak tasks demonstrate that prompts with emojis could easily induce toxicity generation. To understand this phenomenon, we conduct model-level interpretations spanning semantic cognition, sequence generation and tokenization, suggesting that emojis can act as a heterogeneous semantic channel to bypass the safety mechanisms. To pursue deeper insights, we further probe the pre-training corpus and uncover potential correlation between the emoji-related data polution with the toxicity generation behaviors. Supplementary materials provide our implementation code and data. (Warning: This paper contains potentially sensitive contents)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11127v1">Joint Effects of Argumentation Theory, Audio Modality and Data Enrichment on LLM-Based Fallacy Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      This study investigates how context and emotional tone metadata influence large language model (LLM) reasoning and performance in fallacy classification tasks, particularly within political debate settings. Using data from U.S. presidential debates, we classify six fallacy types through various prompting strategies applied to the Qwen-3 (8B) model. We introduce two theoretically grounded Chain-of-Thought frameworks: Pragma-Dialectics and the Periodic Table of Arguments, and evaluate their effectiveness against a baseline prompt under three input settings: text-only, text with context, and text with both context and audio-based emotional tone metadata. Results suggest that while theoretical prompting can improve interpretability and, in some cases, accuracy, the addition of context and especially emotional tone metadata often leads to lowered performance. Emotional tone metadata biases the model toward labeling statements as \textit{Appeal to Emotion}, worsening logical reasoning. Overall, basic prompts often outperformed enhanced ones, suggesting that attention dilution from added inputs may worsen rather than improve fallacy classification in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v4">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11079v1">Difficulty-Aware Agent Orchestration in LLM-Powered Workflows</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or task-level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailoring workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11076v1">Chameleon: Taming Dynamic Operator Sequences for Memory-Intensive LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The increasing size of large language models (LLMs) has led to a surge in memory requirements during training, often exceeding the capacity of high-bandwidth memory (HBM). Swap-based memory optimization incurs neither accuracy loss nor additional end-to-end overhead when effectively overlapped, thus being an attractive solution. However, existing swap methods assume consistent operator sequences, which is impractical in Eager Mode, where operator sequences can vary during change. We propose Chameleon, which redesigns the end-to-end process of swap-based memory optimization and is the first work to consider varying operator sequences in Eager Mode. Chameleon (i) introduces a lightweight online profiler to enable continuous profiling for monitoring operator sequences, (ii) generates effective swap policies with limited operator information, and (iii) optimizes the policy execution module for accurate policy application and better performance. Experimental results demonstrate that Chameleon reduces profiling overhead by 84.25%, enables training models up to 4x larger than hardware memory while adapting to changes in operator sequences, improves performance by up to 38.94% compared to recomputation or high-degree parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11026v1">Rethinking Human Preference Evaluation of LLM Rationales</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Published in the XLLM-Reason-Plan Workshop on the Application of LLM Explainability to Reasoning and Planning at COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12283v1">Prompting the Professoriate: A Qualitative Study of Instructor Perspectives on LLMs in Data Science Education</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 45 pages, 16 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shifted in just a few years from novelty to ubiquity, raising fundamental questions for data science education. Tasks once used to teach coding, writing, and problem-solving can now be completed by LLMs, forcing educators to reconsider both pedagogy and assessment. To understand how instructors are adapting, we conducted semi-structured interviews with 42 instructors from 33 institutions in 10 countries in June and July 2025. Our qualitative analysis reveals a pragmatic mix of optimism and concern. Many respondents view LLMs as inevitable classroom tools -- comparable to calculators or Wikipedia -- while others worry about de-skilling, misplaced confidence, and uneven integration across institutions. Around 58 per cent have already introduced demonstrations, guided activities, or make extensive use of LLMs in their courses, though most expect change to remain slow and uneven. That said, 31 per cent have not used LLMs to teach students and do not plan to. We highlight some instructional innovations, including AI-aware assessments, reflective use of LLMs as tutors, and course-specific chatbots. By sharing these perspectives, we aim to help data science educators adapt collectively to ensure curricula keep pace with technological change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12273v1">LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has made natural language-driven route planning an emerging research area that encompasses rich user objectives. Current research exhibits two distinct approaches: direct route planning using LLM-as-Agent and graph-based searching strategies. However, LLMs in the former approach struggle to handle extensive map data, while the latter shows limited capability in understanding natural language preferences. Additionally, a more critical challenge arises from the highly heterogeneous and unpredictable spatio-temporal distribution of users across the globe. In this paper, we introduce a novel LLM-Assisted route Planning (LLMAP) system that employs an LLM-as-Parser to comprehend natural language, identify tasks, and extract user preferences and recognize task dependencies, coupled with a Multi-Step Graph construction with iterative Search (MSGS) algorithm as the underlying solver for optimal route finding. Our multi-objective optimization approach adaptively tunes objective weights to maximize points of interest (POI) quality and task completion rate while minimizing route distance, subject to three key constraints: user time limits, POI opening hours, and task dependencies. We conduct extensive experiments using 1,000 routing prompts sampled with varying complexity across 14 countries and 27 cities worldwide. The results demonstrate that our approach achieves superior performance with guarantees across multiple constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13352v1">Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 14 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13351v1">Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11466v1">Improving LLMs' Learning for Coreference Resolution</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is crucial for many NLP tasks, but existing LLMs struggle with hallucination and under-performance. In this paper, we investigate the limitations of existing LLM-based approaches to CR-specifically the Question-Answering (QA) Template and Document Template methods and propose two novel techniques: Reversed Training with Joint Inference and Iterative Document Generation. Our experiments show that Reversed Training improves the QA Template method, while Iterative Document Generation eliminates hallucinations in the generated source text and boosts coreference resolution. Integrating these methods and techniques offers an effective and robust solution to LLM-based coreference resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11446v1">Large Language Models (LLMs) for Requirements Engineering (RE): A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are finding applications in numerous domains, and Requirements Engineering (RE) is increasingly benefiting from their capabilities to assist with complex, language-intensive tasks. This paper presents a systematic literature review of 74 primary studies published between 2023 and 2024, examining how LLMs are being applied in RE. The study categorizes the literature according to several dimensions, including publication trends, RE activities, prompting strategies, and evaluation methods. Our findings indicate notable patterns, among which we observe substantial differences compared to previous works leveraging standard Natural Language Processing (NLP) techniques. Most of the studies focus on using LLMs for requirements elicitation and validation, rather than defect detection and classification, which were dominant in the past. Researchers have also broadened their focus and addressed novel tasks, e.g., test generation, exploring the integration of RE with other software engineering (SE) disciplines. Although requirements specifications remain the primary focus, other artifacts are increasingly considered, including issues from issue tracking systems, regulations, and technical manuals. The studies mostly rely on GPT-based models, and often use Zero-shot or Few-shot prompting. They are usually evaluated in controlled environments, with limited use in industry settings and limited integration in complex workflows. Our study outlines important future directions, such as leveraging the potential to expand the influence of RE in SE, exploring less-studied tasks, improving prompting methods, and testing in real-world environments. Our contribution also helps researchers and practitioners use LLMs more effectively in RE, by providing a list of identified tools leveraging LLMs for RE, as well as datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11420v1">Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Tauric Research: https://github.com/TauricResearch
    </div>
    <details class="paper-abstract">
      Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at https://github.com/TauricResearch/Trading-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04249v2">IOLBENCH: Benchmarking LLMs on Linguistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Despite the remarkable advancements and widespread applications of deep neural networks, their ability to perform reasoning tasks remains limited, particularly in domains requiring structured, abstract thought. In this paper, we investigate the linguistic reasoning capabilities of state-of-the-art large language models (LLMs) by introducing IOLBENCH, a novel benchmark derived from International Linguistics Olympiad (IOL) problems. This dataset encompasses diverse problems testing syntax, morphology, phonology, and semantics, all carefully designed to be self-contained and independent of external knowledge. These tasks challenge models to engage in metacognitive linguistic reasoning, requiring the deduction of linguistic rules and patterns from minimal examples. Through extensive benchmarking of leading LLMs, we find that even the most advanced models struggle to handle the intricacies of linguistic complexity, particularly in areas demanding compositional generalization and rule abstraction. Our analysis highlights both the strengths and persistent limitations of current models in linguistic problem-solving, offering valuable insights into their reasoning capabilities. By introducing IOLBENCH, we aim to foster further research into developing models capable of human-like reasoning, with broader implications for the fields of computational linguistics and artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21094v2">Video-LLMs with Temporal Visual Screening</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Humans naturally perform temporal screening by dragging the progress bar and focusing on salient temporal segments, but current Video Large Language Models (Video-LLMs) struggle to capture fine-grained temporal semantics due to sparse frame sampling and insufficient inter-frame reasoning supervision during their training. To address this, Inspired by well-established cognitive science principles, we propose Temporal Visual Screening (TVS), a new task that universally pre-processes video question answering and instruction tuning data by: (1) retaining focus-critical video segments, (2) synchronously reconstructing queries to their most direct form while preserving answer consistency, and (3) keeping the invariance and consistency for any possible answer. TVS is formulated as a modular front-end adapter task that can be seamlessly integrated into both Video Instruction Tuning (training) and Video Question Answering (inference) pipelines. TVS optimizes distribution of reasoning burden and cognitive load; during training, it aligns queries with focus-critical visual information; at inference, it enables query-aware segment focus and streamlined query representations. In particular, we curate the first benchmark for TVS and propose ReSimplifyIt, a baseline outperforming prior approaches on seemingly similar tasks by 0.47 in F-1 score on video trimming while achieving competitive query rewriting performance. Experiments demonstrate that incorporating TVS yields relative gains of 7.33% (training) and 34.6% (inference), demonstrating the effectiveness of temporal information screening for improving video-language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10975v1">ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 CCKS 2025 Shared Task Paper
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07238v2">Can Advanced LLMs Coach Smaller LLMs? Knowledge Distillation for Goal-Oriented Dialogs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Enterprises deploying LLMs for goal-oriented dialogs, such as customer service, face a critical trade-off between performance, control, and cost. Proprietary models like GPT-4 offer strong performance but are costly and cannot be self-hosted, raising security and privacy concerns. Open-source alternatives offer flexibility and lower token costs but lag in performance. We introduce Guidance Elicitation and Retrieval (GER), a prompt-based knowledge distillation framework where a high-performance teacher LLM coaches a lower-performance student without modifying the student's parameters. GER extracts tactical guidance for a wide range of dialog scenarios from the teacher and stores these scenario-guidance pairs in a structured library. At inference time, the student retrieves the relevant guidance and integrates it into its prompt. While GER training can be bootstrapped entirely with synthetic data, its modular design lets it seamlessly augment the synthetic data with human conversational logs. In addition, the modular design enables easy auditing and updating of the guidance library as new scenarios and constraints emerge. Experiments show GER's guidance-based coaching outperforms both example output based fine-tuning and non-customized guidance baselines, and generalizes across other contexts and student models. The GER framework is potentially extensible to coach human service agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10972v1">Enhancing Computational Cognitive Architectures with LLMs: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Computational cognitive architectures are broadly scoped models of the human mind that combine different psychological functionalities (as well as often different computational methods for these different functionalities) into one unified framework. They structure them in a psychologically plausible and validated way. However, such models thus far have only limited computational capabilities, mostly limited by the computational tools and techniques that were adopted. More recently, LLMs have proved to be more capable computationally than any other tools. Thus, in order to deal with both real-world complexity and psychological realism at the same time, incorporating LLMs into cognitive architectures naturally becomes an important task. In the present article, a synergistic combination of the Clarion cognitive architecture and LLMs is discussed as a case study. The implicit-explicit dichotomy that is fundamental to Clarion is leveraged for a seamless integration of Clarion and LLMs. As a result, computational power of LLMs is combined with psychological nicety of Clarion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10946v1">When the Code Autopilot Breaks: Why LLMs Falter in Embedded Machine Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 This paper has been accepted for publication in Computer (IEEE). Upon publication, the copyright will be transferred to IEEE
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate software generation in embedded machine learning workflows, yet their outputs often fail silently or behave unpredictably. This article presents an empirical investigation of failure modes in LLM-powered ML pipelines, based on an autopilot framework that orchestrates data preprocessing, model conversion, and on-device inference code generation. We show how prompt format, model behavior, and structural assumptions influence both success rates and failure characteristics, often in ways that standard validation pipelines fail to detect. Our analysis reveals a diverse set of error-prone behaviors, including format-induced misinterpretations and runtime-disruptive code that compiles but breaks downstream. We derive a taxonomy of failure categories and analyze errors across multiple LLMs, highlighting common root causes and systemic fragilities. Though grounded in specific devices, our study reveals broader challenges in LLM-based code generation. We conclude by discussing directions for improving reliability and traceability in LLM-powered embedded ML systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22169v2">ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10931v1">Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10860v1">Quantifier Scope Interpretation in Language Learners and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Sentences with multiple quantifiers often lead to interpretive ambiguities, which can vary across languages. This study adopts a cross-linguistic approach to examine how large language models (LLMs) handle quantifier scope interpretation in English and Chinese, using probabilities to assess interpretive likelihood. Human similarity (HS) scores were used to quantify the extent to which LLMs emulate human performance across language groups. Results reveal that most LLMs prefer the surface scope interpretations, aligning with human tendencies, while only some differentiate between English and Chinese in the inverse scope preferences, reflecting human-similar patterns. HS scores highlight variability in LLMs' approximation of human behavior, but their overall potential to align with humans is notable. Differences in model architecture, scale, and particularly models' pre-training data language background, significantly influence how closely LLMs approximate human quantifier scope interpretations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06646v2">Evaluating and Aligning Human Economic Risk Preferences in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10830v1">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10818v1">LLM Enhancement with Domain Expert Mental Model to Reduce LLM Hallucination with Causal Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 25 pages,4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Difficult decision-making problems abound in various disciplines and domains. The proliferation of generative techniques, especially large language models (LLMs), has excited interest in using them for decision support. However, LLMs cannot yet resolve missingness in their training data, leading to hallucinations. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external information retrieval, reducing hallucinations and improving accuracy. Yet, RAG and related methods are only partial solutions, as they may lack access to all necessary sources or key missing information. Even everyday issues often challenge LLMs' abilities. Submitting longer prompts with context and examples is one approach to address knowledge gaps, but designing effective prompts is non-trivial and may not capture complex mental models of domain experts. For tasks with missing critical information, LLMs are insufficient, as are many existing systems poorly represented in available documents. This paper explores how LLMs can make decision-making more efficient, using a running example of evaluating whether to respond to a call for proposals. We propose a technology based on optimized human-machine dialogue and monotone Boolean and k-valued functions to discover a computationally tractable personal expert mental model (EMM) of decision-making. Our EMM algorithm for LLM prompt engineering has four steps: (1) factor identification, (2) hierarchical structuring of factors, (3) generating a generalized expert mental model specification, and (4) generating a detailed generalized expert mental model from that specification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06770v2">Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative iteration. Together, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v3">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with multiple golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight the performance gap between open-source models and top students, the strong reasoning abilities of closed-source models, and the remaining room for improvement. HiPhO, a human-aligned Olympiad benchmark for multimodal physical reasoning, is open-source at https://github.com/SciYu/HiPhO with a public leaderboard at https://phyarena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10625v1">No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10594v1">SME-TEAM: Leveraging Trust and Ethics for Secure and Responsible Use of AI and LLMs in SMEs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and Large Language Models (LLMs) are reshaping today's business practices, however, their adoption within small and medium-sized enterprises (SMEs) raises significant technical, ethical and trust issues. This paper proposes a structured, multi-phased framework designed to embed trust and ethical principles throughout the AI lifecycle for their secure and responsible use in SMEs. Structured around four pillars, i.e., Data, Algorithms, Human oversight, and Model Architecture, the framework bridges theoretical ethical principles with operational practice, enhancing AI capabilities in diverse SME applications. Ultimately, this paper offers a structured roadmap for responsible AI adoption, framing trust and ethics as a catalyst for resilience, competitiveness, and sustainable innovation in SMEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10436v1">RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 12 pages, 5 figures, submitted to IEEE Transactions on Services Computing
    </div>
    <details class="paper-abstract">
      To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured, multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of Large Language Models (LLMs) across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers various technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges from Stack Overflow. Extensive experiments reveal that our fine-tuned model, RefactorCoder-MoE, achieves state-of-the-art performance, significantly outperforming leading open-source and commercial baselines with an overall accuracy of 76.84%. Human evaluations further validate the interpretability, accuracy, and practical relevance of the generated solutions. In addition, we evaluate system-level metrics, such as throughput and latency, to gain deeper insights into the performance characteristics and trade-offs of the proposed architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18889v2">Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      NLP benchmarks rely on standardized datasets for training and evaluating models and are crucial for advancing the field. Traditionally, expert annotations ensure high-quality labels; however, the cost of expert annotation does not scale well with the growing demand for larger datasets required by modern models. While crowd-sourcing provides a more scalable solution, it often comes at the expense of annotation precision and consistency. Recent advancements in large language models (LLMs) offer new opportunities to enhance the annotation process, particularly for detecting label errors in existing datasets. In this work, we consider the recent approach of LLM-as-a-judge, leveraging an ensemble of LLMs to flag potentially mislabeled examples. We conduct a case study on four factual consistency datasets from the TRUE benchmark, spanning diverse NLP tasks, and on SummEval, which uses Likert-scale ratings of summary quality across multiple dimensions. We empirically analyze the labeling quality of existing datasets and compare expert, crowd-sourced, and LLM-based annotations in terms of the agreement, label quality, and efficiency, demonstrating the strengths and limitations of each annotation method. Our findings reveal a substantial number of label errors, which, when corrected, induce a significant upward shift in reported model performance. This suggests that many of the LLMs' so-called mistakes are due to label errors rather than genuine model failures. Additionally, we discuss the implications of mislabeled data and propose methods to mitigate them in training to improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10402v1">Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming integral to modern software development workflows, assisting developers with code generation, API explanation, and iterative problem-solving through natural language conversations. Despite widespread adoption, there is limited understanding of how developers interact with LLMs in practice and how these conversational dynamics influence task outcomes, code quality, and software engineering workflows. To address this, we leverage CodeChat, a large dataset comprising 82,845 real-world developer-LLM conversations, containing 368,506 code snippets generated across over 20 programming languages, derived from the WildChat dataset. We find that LLM responses are substantially longer than developer prompts, with a median token-length ratio of 14:1. Multi-turn conversations account for 68% of the dataset and often evolve due to shifting requirements, incomplete prompts, or clarification requests. Topic analysis identifies web design (9.6% of conversations) and neural network training (8.7% of conversations) as the most frequent LLM-assisted tasks. Evaluation across five languages (i.e., Python, JavaScript, C++, Java, and C#) reveals prevalent and language-specific issues in LLM-generated code: generated Python and JavaScript code often include undefined variables (83.4% and 75.3% of code snippets, respectively); Java code lacks required comments (75.9%); C++ code frequently omits headers (41.1%) and C# code shows unresolved namespaces (49.2%). During a conversation, syntax and import errors persist across turns; however, documentation quality in Java improves by up to 14.7%, and import handling in Python improves by 3.7% over 5 turns. Prompts that point out mistakes in code generated in prior turns and explicitly request a fix are most effective for resolving errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02896v2">Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 To appear in SIGMOD'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly used as a building block in data systems to process large text datasets. To do so, LLM model providers offer multiple LLMs with different sizes, spanning various cost-quality trade-offs when processing text at scale. Top-of-the-line LLMs (e.g., GPT-4o, Claude Sonnet) operate with high accuracy but are prohibitively expensive when processing many records. To avoid high costs, more affordable but lower quality LLMs (e.g., GPT-4o-mini, Claude Haiku) can be used to process records, but we need to ensure that the overall accuracy does not deviate substantially from that of the top-of-the-line LLMs. The model cascade framework provides a blueprint to manage this trade-off, by using the confidence of LLMs in their output (e.g., log-probabilities) to decide on which records to use the affordable LLM. However, existing solutions following this framework provide only marginal cost savings and weak theoretical guarantees because of poor estimation of the quality of the affordable LLM's outputs. We present BARGAIN, a method that judiciously uses affordable LLMs in data processing to significantly reduce cost while providing strong theoretical guarantees on the solution quality. BARGAIN employs a novel adaptive sampling strategy and statistical estimation procedure that uses data and task characteristics and builds on recent statistical tools to make accurate estimations with tight theoretical guarantees. Variants of BARGAIN can support guarantees on accuracy, precision, or recall of the output. Experimental results across 8 real-world datasets show that BARGAIN reduces cost, on average, by up to 86% more than state-of-the-art, while providing stronger theoretical guarantees on accuracy of output, with similar gains when guaranteeing a desired level of precision or recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10377v1">Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted to EMNLP2025
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (SMoE) architectures are widely used in large language models (LLMs) due to their computational efficiency. However, though only a few experts are activated for each token, SMoE still requires loading all expert parameters, leading to high memory usage and challenges in deployment. Previous work has tried to reduce the overhead by pruning and merging experts, but primarily focused on expert-level operations, leaving neuron-level structure underexplored. We propose DERN (Dropping Experts, Recombining Neurons), a task-agnostic and retraining-free framework for expert pruning and reconstruction. We observe that experts are often misaligned and contain semantic conflicts at the neuron level, which poses challenges for direct merging. To solve this, DERN works in three steps: it first prunes redundant experts using router statistics; then it decomposes them into neuron-level expert segments, assigning each segment to its most compatible retained expert; and finally, it merges segments within each retained expert to build a compact representation. Experiments on Mixtral, Qwen, and DeepSeek SMoE models show that DERN improves performance by more than 5% on commonsense reasoning and MMLU benchmarks under 50% expert sparsity, without extra training. It also greatly reduces the number of experts and memory usage, making SMoE LLMs easier to deploy in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10372v1">MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant inference latency due to inefficiencies in GEMM operations, weight access, and KV cache access, especially in real-time scenarios. This highlights the need for a versatile compute-memory efficient accelerator. Unfortunately, existing Transformer accelerators struggle to address both aspects simultaneously, as they focus on value-level processing, missing fine-grained opportunities to optimize computation and memory collaboratively. This paper introduces MCBP, a bit-grained compute-memory efficient algorithm-hardware co-design that leverages bit-slice (BS) enabled repetitiveness and sparsity to accelerate LLM inference. MCBP features three key innovations: 1) BS-repetitiveness-enabled computation reduction (BRCR), which eliminates redundant GEMM computations via leveraging redundancy hidden among BS vectors; 2) BS-sparsity-enabled two-state coding (BSTC), which reduces weight access via exploiting significant sparsity in high-order bit-slice weight; 3) Bit-grained progressive prediction (BGPP), which reduces KV cache access by leveraging early-termination-based bit-grained prediction. These techniques, supported by custom accelerator designs, effectively alleviate the burden in GEMM, weight access, and KV cache access. Extensive experiments on 26 benchmarks show that MCBP achieves 9.43x speed up and 31.1x higher energy efficiency than Nvidia A100 GPU. Compared to SOTA Transformer accelerators, MCBP achieves 35x, 5.2x and 3.2x energy saving than Spatten, FACT and SOFA, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18173v3">UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 This article was not accepted, and its quality is not very good. Therefore, we have decided to withdraw the submission and will not resubmit it elsewhere
    </div>
    <details class="paper-abstract">
      Managing long texts is challenging for large language models (LLMs) due to limited context window sizes. This study introduces UIO-LLMs, an unbiased incremental optimization approach for memory-enhanced transformers under long-context settings. We initially conceptualize the process as a streamlined encoder-decoder framework where the weights-shared encoder and decoder respectively encapsulate a context segment into memories and leverage these memories to predict outputs of the subsequent segment. Subsequently, by treating our memory-enhanced transformers as fully-connected recurrent neural networks (RNNs), we refine the training process using the Truncated Backpropagation Through Time (TBPTT) algorithm, which incorporates innovative incremental optimization techniques. These techniques not only diminish time complexity but also address the bias in gradient computation through an unbiased optimization process. UIO-LLMs successfully handle long context, such as extending the context window of Llama2-7b-chat from 4K to 100K tokens with minimal 2% additional parameters, while keeping the inference cost nearly linear as context length increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10297v1">The Morality of Probability: How Implicit Moral Biases in LLMs May Shape the Future of Human-AI Symbiosis</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is advancing at a pace that raises urgent questions about how to align machine decision-making with human moral values. This working paper investigates how leading AI systems prioritize moral outcomes and what this reveals about the prospects for human-AI symbiosis. We address two central questions: (1) What moral values do state-of-the-art large language models (LLMs) implicitly favour when confronted with dilemmas? (2) How do differences in model architecture, cultural origin, and explainability affect these moral preferences? To explore these questions, we conduct a quantitative experiment with six LLMs, ranking and scoring outcomes across 18 dilemmas representing five moral frameworks. Our findings uncover strikingly consistent value biases. Across all models, Care and Virtue values outcomes were rated most moral, while libertarian choices were consistently penalized. Reasoning-enabled models exhibited greater sensitivity to context and provided richer explanations, whereas non-reasoning models produced more uniform but opaque judgments. This research makes three contributions: (i) Empirically, it delivers a large-scale comparison of moral reasoning across culturally distinct LLMs; (ii) Theoretically, it links probabilistic model behaviour with underlying value encodings; (iii) Practically, it highlights the need for explainability and cultural awareness as critical design principles to guide AI toward a transparent, aligned, and symbiotic future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13335v2">Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Humour, as a complex language form, is derived from myriad aspects of life. Whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular form. We compare models' joke explanation abilities from simple puns to complex topical humour that requires esoteric knowledge of real-world entities and events. To this end, we curate a dataset of 600 jokes across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (including reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most existing works on overly simple joke forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v2">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v1">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10179v1">Benchmark of stylistic variation in LLM-generated texts</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v3">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to build autonomous agents that perform complex tasks with external tools, often exposed through APIs in enterprise systems. Direct use of these APIs is difficult due to the complex input schema and verbose responses. Current benchmarks overlook these challenges, leaving a gap in assessing API readiness for agent-driven automation. We present a testing framework that systematically evaluates enterprise APIs when wrapped as Python tools for LLM-based agents. The framework generates data-aware test cases, translates them into natural language instructions, and evaluates whether agents can correctly invoke the tool, handle their inputs, and process its responses. We apply the framework to generate over 2400 test cases across different domains and develop a taxonomy of common errors, including input misinterpretation, output failures, and schema mismatches. We further classify errors to support debugging and tool refinement. Our framework provides a systematic approach to enabling enterprise APIs as reliable tools for agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05367v2">Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10127v1">Population-Aligned Persona Generation for LLM-based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08863v2">GeoJSON Agents:A Multi-Agent LLM Architecture for Geospatial Analysis-Function Calling vs Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      LLMs have made substantial progress in task automation and natural language understanding. However, without expertise in GIS, they continue to encounter limitations. To address these issues, we propose GeoJSON Agents-a multi-agent LLM architecture. This framework transforms natural language tasks into structured GeoJSON operation commands and processes spatial data using two widely adopted LLM enhancement techniques: Function Calling and Code Generation. The architecture consists of three components-task parsing, agent collaboration, and result integration-aimed at enhancing both the performance and scalability of GIS automation. The Planner agent interprets natural language tasks into structured GeoJSON commands. Then, specialized Worker agents collaborate according to assigned roles to perform spatial data processing and analysis, either by invoking predefined function APIs or by dynamically generating and executing Python-based spatial analysis code. Finally, the system integrates the outputs from multiple execution rounds into reusable, standards-compliant GeoJSON files. To systematically evaluate the performance of the two approaches, we constructed a benchmark dataset of 70 tasks with varying complexity and conducted experiments using OpenAI's GPT-4o as the core model. Results indicate that the Function Calling-based GeoJSON Agent achieved an accuracy of 85.71%, while the Code Generation-based agent reached 97.14%, both significantly outperforming the best-performing general-purpose model (48.57%). Further analysis reveals that the Code Generation provides greater flexibility, whereas the Function Calling approach offers more stable execution. This study is the first to introduce an LLM multi-agent framework for GeoJSON data and to compare the strengths and limitations of two mainstream LLM enhancement methods, offering new perspectives for improving GeoAI system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10010v1">Multi-Intent Recognition in Dialogue Understanding: A Comparison Between Smaller Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      In this paper, we provide an extensive analysis of multi-label intent classification using Large Language Models (LLMs) that are open-source, publicly available, and can be run in consumer hardware. We use the MultiWOZ 2.1 dataset, a benchmark in the dialogue system domain, to investigate the efficacy of three popular open-source pre-trained LLMs, namely LLama2-7B-hf, Mistral-7B-v0.1, and Yi-6B. We perform the classification task in a few-shot setup, giving 20 examples in the prompt with some instructions. Our approach focuses on the differences in performance of these models across several performance metrics by methodically assessing these models on multi-label intent classification tasks. Additionally, we compare the performance of the instruction-based fine-tuning approach with supervised learning using the smaller transformer model BertForSequenceClassification as a baseline. To evaluate the performance of the models, we use evaluation metrics like accuracy, precision, and recall as well as micro, macro, and weighted F1 score. We also report the inference time, VRAM requirements, etc. The Mistral-7B-v0.1 outperforms two other generative models on 11 intent classes out of 14 in terms of F-Score, with a weighted average of 0.50. It also has relatively lower Humming Loss and higher Jaccard Similarity, making it the winning model in the few-shot setting. We find BERT based supervised classifier having superior performance compared to the best performing few-shot generative LLM. The study provides a framework for small open-source LLMs in detecting complex multi-intent dialogues, enhancing the Natural Language Understanding aspect of task-oriented chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09995v1">QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated impressive capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks, leveraging fundamental and sentiment-based inputs for strategic decision-making. However, such systems are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT requires rapid, risk-aware decisions based on structured, short-horizon signals, including technical indicators, chart patterns, and trend-based features, distinct from the long-term semantic reasoning typical of traditional financial LLM applications. To this end, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents, Indicator, Pattern, Trend, and Risk, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. In zero-shot evaluations across ten financial instruments, including Bitcoin and Nasdaq futures, QuantAgent demonstrates superior performance in both predictive accuracy and cumulative return over 4-hour trading intervals, outperforming strong neural and rule-based baselines. Our findings suggest that combining structured financial priors with language-native reasoning unlocks new potential for traceable, real-time decision systems in high-frequency financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09970v1">Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise in generating firmware for embedded systems, but often introduce security flaws and fail to meet real-time performance constraints. This paper proposes a three-phase methodology that combines LLM-based firmware generation with automated security validation and iterative refinement in a virtualized environment. Using structured prompts, models like GPT-4 generate firmware for networking and control tasks, deployed on FreeRTOS via QEMU. These implementations are tested using fuzzing, static analysis, and runtime monitoring to detect vulnerabilities such as buffer overflows (CWE-120), race conditions (CWE-362), and denial-of-service threats (CWE-400). Specialized AI agents for Threat Detection, Performance Optimization, and Compliance Verification collaborate to improve detection and remediation. Identified issues are categorized using CWE, then used to prompt targeted LLM-generated patches in an iterative loop. Experiments show a 92.4\% Vulnerability Remediation Rate (37.3\% improvement), 95.8\% Threat Model Compliance, and 0.87 Security Coverage Index. Real-time metrics include 8.6ms worst-case execution time and 195{\mu}s jitter. This process enhances firmware security and performance while contributing an open-source dataset for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11007v3">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, together with their large model size, make their deployment more challenging. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design TMO, a local-cloud LLM inference system with Three-M Offloading: Multi-modal, Multi-task, and Multi-dialogue. TMO incorporates (i) a lightweight local LLM that can process simple tasks at high speed and (ii) a large-scale cloud LLM that can handle multi-modal data sources. We develop a resource-constrained reinforcement learning (RCRL) strategy for TMO that optimizes the inference location (i.e., local vs. cloud) and multi-modal data sources to use for each task/dialogue, aiming to maximize the long-term reward (response quality, latency, and usage cost) while adhering to resource constraints. We also contribute M4A1, a new dataset we curated that contains reward and cost metrics across multiple modality, task, dialogue, and LLM configurations, enabling evaluation of offloading decisions. We demonstrate the effectiveness of TMO compared to several exploration-decision and LLM-as-Agent baselines, showing significant improvements in latency, cost, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03703v2">Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted at ACM RecSys 2025 (10 pages, 4 figures)
    </div>
    <details class="paper-abstract">
      The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09928v1">Fraud detection and risk assessment of online payment transactions on e-commerce platforms based on LLM and GCN frameworks</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      With the rapid growth of e-commerce, online payment fraud has become increasingly complex, posing serious threats to financial security and consumer trust. Traditional detection methods often struggle to capture the intricate relational structures inherent in transactional data. This study presents a novel fraud detection framework that combines Large Language Models (LLM) with Graph Convolutional Networks (GCN) to effectively identify fraudulent activities in e-commerce online payment transactions. A dataset of 2,840,000 transactions was collected over 14 days from major platforms such as Amazon, involving approximately 2,000 U.S.-based consumers and 30 merchants. With fewer than 6000 fraudulent instances, the dataset represents a highly imbalanced scenario. Consumers and merchants were modeled as nodes and transactions as edges to form a heterogeneous graph, upon which a GCN was applied to learn complex behavioral patterns. Semantic features extracted via GPT-4o and Tabformer were integrated with structural features to enhance detection performance. Experimental results demonstrate that the proposed model achieves an accuracy of 0.98, effectively balancing precision and sensitivity in fraud detection. This framework offers a scalable and real-time solution for securing online payment environments and provides a promising direction for applying graph-based deep learning in financial fraud prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19634v4">Faster and Better LLMs via Latency-Aware Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09917v1">SLD-Spec: Enhancement LLM-assisted Specification Generation for Complex Loop Functions via Program Slicing and Logical Deletion</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 22 pages, 2 figures, conference
    </div>
    <details class="paper-abstract">
      Automatically generating formal specifications from program code can greatly enhance the efficiency of program verification and enable end-to-end automation from requirements to reliable software. However, existing LLM-based approaches often struggle with programs that include complex loop structures, leading to irrelevant specifications. Moreover, the rigorous proof obligations and design constraints imposed by verification tools can further result in incomplete and ambiguous specifications. To address these challenges, we propose SLD-Spec, an LLM-assisted specification generation method tailored for programs with complex loop constructs. SLD-Spec introduces two novel phases into the traditional specification generation framework: (1) A slicing phase, which decomposes each function into code fragments containing independent loop structures, thereby reducing the complexity of specification generation; and (2) A logical deletion phase, which applies LLM-based reasoning to filter out incorrect candidate specifications--especially those not easily identified by verification tool--while retaining valid ones. Experimental results show that on the simple dataset, SLD-Spec successfully verifies five more programs than the state-of-the-art AutoSpec and reduces runtime by 23.73%. To address the limitations of existing research, we manually construct a dataset comprising four categories of complex loop programs. On this dataset, SLD-Spec significantly improves the correctness, relevance, and completeness of generated specifications compared to baseline methods, enabling 95.1% of assertions and 90.91% of programs to pass verification. Ablation studies further reveal that logical deletion is critical for enhancing specification correctness and relevance, while program slicing contributes significantly to specification completeness. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09912v1">When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10753v1">HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning and question-answering capabilities. However, they often produce inaccurate or unreliable content known as hallucinations. This unreliability significantly limits their deployment in high-stakes applications. Thus, there is a growing need for a general-purpose method to detect hallucinations in LLMs. In this work, we introduce HalluField, a novel field-theoretic approach for hallucination detection based on a parametrized variational principle and thermodynamics. Inspired by thermodynamics, HalluField models an LLM's response to a given query and temperature setting as a collection of discrete likelihood token paths, each associated with a corresponding energy and entropy. By analyzing how energy and entropy distributions vary across token paths under changes in temperature and likelihood, HalluField quantifies the semantic stability of a response. Hallucinations are then detected by identifying unstable or erratic behavior in this energy landscape. HalluField is computationally efficient and highly practical: it operates directly on the model's output logits without requiring fine-tuning or auxiliary neural networks. Notably, the method is grounded in a principled physical interpretation, drawing analogies to the first law of thermodynamics. Remarkably, by modeling LLM behavior through this physical lens, HalluField achieves state-of-the-art hallucination detection performance across models and datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10739v1">Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 25 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10729v1">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10723v1">Dark Patterns Meet GUI Agents: LLM Agent Susceptibility to Manipulative Interfaces and the Role of Human Oversight</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      The dark patterns, deceptive interface designs manipulating user behaviors, have been extensively studied for their effects on human decision-making and autonomy. Yet, with the rising prominence of LLM-powered GUI agents that automate tasks from high-level intents, understanding how dark patterns affect agents is increasingly important. We present a two-phase empirical study examining how agents, human participants, and human-AI teams respond to 16 types of dark patterns across diverse scenarios. Phase 1 highlights that agents often fail to recognize dark patterns, and even when aware, prioritize task completion over protective action. Phase 2 revealed divergent failure modes: humans succumb due to cognitive shortcuts and habitual compliance, while agents falter from procedural blind spots. Human oversight improved avoidance but introduced costs such as attentional tunneling and cognitive load. Our findings show neither humans nor agents are uniformly resilient, and collaboration introduces new vulnerabilities, suggesting design needs for transparency, adjustable autonomy, and oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10698v1">CrunchLLM: Multitask LLMs for Structured Business Reasoning and Outcome Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Predicting the success of start-up companies, defined as achieving an exit through acquisition or IPO, is a critical problem in entrepreneurship and innovation research. Datasets such as Crunchbase provide both structured information (e.g., funding rounds, industries, investor networks) and unstructured text (e.g., company descriptions), but effectively leveraging this heterogeneous data for prediction remains challenging. Traditional machine learning approaches often rely only on structured features and achieve moderate accuracy, while large language models (LLMs) offer rich reasoning abilities but struggle to adapt directly to domain-specific business data. We present \textbf{CrunchLLM}, a domain-adapted LLM framework for startup success prediction. CrunchLLM integrates structured company attributes with unstructured textual narratives and applies parameter-efficient fine-tuning strategies alongside prompt optimization to specialize foundation models for entrepreneurship data. Our approach achieves accuracy exceeding 80\% on Crunchbase startup success prediction, significantly outperforming traditional classifiers and baseline LLMs. Beyond predictive performance, CrunchLLM provides interpretable reasoning traces that justify its predictions, enhancing transparency and trustworthiness for financial and policy decision makers. This work demonstrates how adapting LLMs with domain-aware fine-tuning and structured--unstructured data fusion can advance predictive modeling of entrepreneurial outcomes. CrunchLLM contributes a methodological framework and a practical tool for data-driven decision making in venture capital and innovation policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10682v1">LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 37 pages, 8 figures, 13 tables
    </div>
    <details class="paper-abstract">
      The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21587v2">A Cross-Cultural Comparison of LLM-based Public Opinion Simulation: Evaluating Chinese and U.S. Models on Diverse Societies</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      This study evaluates the ability of DeepSeek, an open-source large language model (LLM), to simulate public opinions in comparison to LLMs developed by major tech companies. By comparing DeepSeek-R1 and DeepSeek-V3 with Qwen2.5, GPT-4o, and Llama-3.3 and utilizing survey data from the American National Election Studies (ANES) and the Zuobiao dataset of China, we assess these models' capacity to predict public opinions on social issues in both China and the United States, highlighting their comparative capabilities between countries. Our findings indicate that DeepSeek-V3 performs best in simulating U.S. opinions on the abortion issue compared to other topics such as climate change, gun control, immigration, and services for same-sex couples, primarily because it more accurately simulates responses when provided with Democratic or liberal personas. For Chinese samples, DeepSeek-V3 performs best in simulating opinions on foreign aid and individualism but shows limitations in modeling views on capitalism, particularly failing to capture the stances of low-income and non-college-educated individuals. It does not exhibit significant differences from other models in simulating opinions on traditionalism and the free market. Further analysis reveals that all LLMs exhibit the tendency to overgeneralize a single perspective within demographic groups, often defaulting to consistent responses within groups. These findings highlight the need to mitigate cultural and demographic biases in LLM-driven public opinion modeling, calling for approaches such as more inclusive training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12110v2">Towards LLM Agents for Earth Observation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted at ICML 2025 Workshop TerraBytes
    </div>
    <details class="paper-abstract">
      Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at https://iandrover.github.io/UnivEarth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10637v1">LLMs Homogenize Values in Constructive Arguments on Value-Laden Topics</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to promote prosocial and constructive discourse online. Yet little is known about how they negotiate and shape underlying values when reframing people's arguments on value-laden topics. We conducted experiments with 347 participants from India and the United States, who wrote constructive comments on homophobic and Islamophobic threads, and reviewed human-written and LLM-rewritten versions of these comments. Our analysis shows that LLM systematically diminishes Conservative values while elevating prosocial values such as Benevolence and Universalism. When these comments were read by others, participants opposing same-sex marriage or Islam found human-written comments more aligned with their values, whereas those supportive of these communities found LLM-rewritten versions more aligned with their values. These findings suggest that LLM-driven value homogenization can shape how diverse viewpoints are represented in contentious debates on value-laden topics and may influence the dynamics of online discourse critically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10682v1">LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 37 pages, 8 figures, 13 tables
    </div>
    <details class="paper-abstract">
      The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09234v1">Agentic LLMs for Question Answering over Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Accepted at ACL workshop SemEval 2025
    </div>
    <details class="paper-abstract">
      Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09226v1">Constructing a Question-Answering Simulator through the Distillation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      The question-answering (QA) simulator is a model that mimics real student learning behaviors and predicts their correctness of their responses to questions. QA simulators enable educational recommender systems (ERS) to collect large amounts of training data without interacting with real students, thereby preventing harmful recommendations made by an undertrained ERS from undermining actual student learning. Given the QA history, there are two categories of solutions to predict the correctness, conducting the simulation: (1) LLM-free methods, which apply a traditional sequential model to transfer the QA history into a vector representation first, and make predictions based on the representation; (2) LLM-based methods, which leverage the domain knowledge and reasoning capability of LLM to enhence the prediction. LLM-free methods offer fast inference but generally yield suboptimal performance. In contrast, most LLM-based methods achieve better results, but at the cost of slower inference speed and higher GPU memory consumption. In this paper, we propose a method named LLM Distillation based Simulator (LDSim), which distills domain knowledge and reasoning capability from an LLM to better assist prediction, thereby improving simulation performance. Extensive experiments demonstrate that our LDSim achieves strong results on both the simulation task and the knowledge tracing (KT) task. Our code is publicly available at https://anonymous.4open.science/r/LDSim-05A9.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19740v3">Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09174v1">EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at https://github.com/FreedomIntelligence/EchoX.
    </details>
</div>
