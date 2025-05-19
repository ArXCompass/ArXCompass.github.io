# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01639v4">Moral Alignment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Published at the 13th International Conference on Learning Representations (ICLR'25), Singapore, Apr 2025. https://openreview.net/forum?id=MeGDmZjUXy
    </div>
    <details class="paper-abstract">
      Decision-making agents based on pre-trained Large Language Models (LLMs) are increasingly being deployed across various domains of human activity. While their applications are currently rather specialized, several research efforts are underway to develop more generalist agents. As LLM-based systems become more agentic, their influence on human activity will grow and their transparency will decrease. Consequently, developing effective methods for aligning them to human values is vital. The prevailing practice in alignment often relies on human preference data (e.g., in RLHF or DPO), in which values are implicit, opaque and are essentially deduced from relative preferences over different model outputs. In this work, instead of relying on human feedback, we introduce the design of reward functions that explicitly and transparently encode core human values for Reinforcement Learning-based fine-tuning of foundation agent models. Specifically, we use intrinsic rewards for the moral alignment of LLM agents. We evaluate our approach using the traditional philosophical frameworks of Deontological Ethics and Utilitarianism, quantifying moral rewards for agents in terms of actions and consequences on the Iterated Prisoner's Dilemma (IPD) environment. We also show how moral fine-tuning can be deployed to enable an agent to unlearn a previously developed selfish strategy. Finally, we find that certain moral strategies learned on the IPD game generalize to several other matrix game environments. In summary, we demonstrate that fine-tuning with intrinsic rewards is a promising general solution for aligning LLM agents to human values, and it might represent a more transparent and cost-effective alternative to currently predominant alignment techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v6">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07078v1">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07049v1">DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07027v1">LLM-Augmented Chemical Synthesis and Design Decision Programs</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06972v1">Web Page Classification using LLMs for Crawling Support</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      A web crawler is a system designed to collect web pages, and efficient crawling of new pages requires appropriate algorithms. While website features such as XML sitemaps and the frequency of past page updates provide important clues for accessing new pages, their universal application across diverse conditions is challenging. In this study, we propose a method to efficiently collect new pages by classifying web pages into two types, "Index Pages" and "Content Pages," using a large language model (LLM), and leveraging the classification results to select index pages as starting points for accessing new pages. We construct a dataset with automatically annotated web page types and evaluate our approach from two perspectives: the page type classification performance and coverage of new pages. Experimental results demonstrate that the LLM-based method outperformed baseline methods in both evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06964v1">From Knowledge to Reasoning: Evaluating LLMs for Ionic Liquids Research in Chemical and Biological Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have achieved remarkable performance in diverse general knowledge and reasoning tasks, their utility in the scientific domain of Chemical and Biological Engineering (CBE) is unclear. Hence, it necessitates challenging evaluation benchmarks that can measure LLM performance in knowledge- and reasoning-based tasks, which is lacking. As a foundational step, we empirically measure the reasoning capabilities of LLMs in CBE. We construct and share an expert-curated dataset of 5,920 examples for benchmarking LLMs' reasoning capabilities in the niche domain of Ionic Liquids (ILs) for carbon sequestration, an emergent solution to reducing global warming. The dataset presents different difficulty levels by varying along the dimensions of linguistic and domain-specific knowledge. Benchmarking three less than 10B parameter open-source LLMs on the dataset suggests that while smaller general-purpose LLMs are knowledgeable about ILs, they lack domain-specific reasoning capabilities. Based on our results, we further discuss considerations for leveraging LLMs for carbon capture research using ILs. Since LLMs have a high carbon footprint, gearing them for IL research can symbiotically benefit both fields and help reach the ambitious carbon neutrality target by 2050. Dataset link: https://github.com/sougata-ub/llms_for_ionic_liquids
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06912v1">Building a Human-Verified Clinical Reasoning Dataset via a Human LLM Hybrid Pipeline for Trustworthy Medical AI</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Despite strong performance in medical question-answering, the clinical adoption of Large Language Models (LLMs) is critically hampered by their opaque 'black-box' reasoning, limiting clinician trust. This challenge is compounded by the predominant reliance of current medical LLMs on corpora from scientific literature or synthetic data, which often lack the granular expert validation and high clinical relevance essential for advancing their specialized medical capabilities. To address these critical gaps, we introduce a highly clinically relevant dataset with 31,247 medical question-answer pairs, each accompanied by expert-validated chain-of-thought (CoT) explanations. This resource, spanning multiple clinical domains, was curated via a scalable human-LLM hybrid pipeline: LLM-generated rationales were iteratively reviewed, scored, and refined by medical experts against a structured rubric, with substandard outputs revised through human effort or guided LLM regeneration until expert consensus. This publicly available dataset provides a vital source for the development of medical LLMs that capable of transparent and verifiable reasoning, thereby advancing safer and more interpretable AI in medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06901v1">Ecco: Improving Memory Bandwidth and Capacity for LLMs via Entropy-aware Cache Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 ISCA 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated transformative capabilities across diverse artificial intelligence applications, yet their deployment is hindered by substantial memory and computational demands, especially in resource-constrained environments. Quantization techniques have emerged as a critical solution, reducing data precision to enhance memory and computational efficiency. However, existing methods often suffer from high runtime overheads and potential accuracy degradation. To address these challenges, we propose Ecco, an entropy-based cache compression technique tailored for LLMs. Ecco combines group-wise and non-uniform quantization with pre-defined shared k-means patterns and Huffman coding to exploit the inherent entropy characteristics of LLM cache data. Recognizing the inefficiencies of traditional Huffman coding in terms of parallelism and latency, we introduce a novel parallel Huffman-based decoding process with a multi-stage pipeline design, reducing latency by two orders of magnitude and achieving throughput comparable to GPU L2 caches. Comprehensive evaluations demonstrate that Ecco achieves an up to 2.9$\times$ and 1.9$\times$ speedup over the state-of-the-art AWQ and SmoothQuant framework, 2.4$\times$ over the Olive accelerator, all while increasing memory capacity by nearly 4$\times$ and maintaining state-of-the-art LLM accuracy. These results underscore the effectiveness of our entropy-based cache compression in enhancing LLM performance and efficiency, paving the way for more deployable large-scale AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10424v5">CodeV: Empowering LLMs with HDL Generation through Multi-Level Summarization</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 13 pages, 10 figures, journal
    </div>
    <details class="paper-abstract">
      The design flow of processors, particularly in hardware description languages (HDL) like Verilog and Chisel, is complex and costly. While recent advances in large language models (LLMs) have significantly improved coding tasks in software languages such as Python, their application in HDL generation remains limited due to the scarcity of high-quality HDL data. Traditional methods of adapting LLMs for hardware design rely on synthetic HDL datasets, which often suffer from low quality because even advanced LLMs like GPT perform poorly in the HDL domain. Moreover, these methods focus solely on chat tasks and the Verilog language, limiting their application scenarios. In this paper, we observe that: (1) HDL code collected from the real world is of higher quality than code generated by LLMs. (2) LLMs like GPT-3.5 excel in summarizing HDL code rather than generating it. (3) An explicit language tag can help LLMs better adapt to the target language when there is insufficient data. Based on these observations, we propose an efficient LLM fine-tuning pipeline for HDL generation that integrates a multi-level summarization data synthesis process with a novel Chat-FIM-Tag supervised fine-tuning method. The pipeline enhances the generation of HDL code from natural language descriptions and enables the handling of various tasks such as chat and infilling incomplete code. Utilizing this pipeline, we introduce CodeV, a series of HDL generation LLMs. Among them, CodeV-All not only possesses a more diverse range of language abilities, i.e. Verilog and Chisel, and a broader scope of tasks, i.e. Chat and fill-in-middle (FIM), but it also achieves performance on VerilogEval that is comparable to or even surpasses that of CodeV-Verilog fine-tuned on Verilog only, making them the first series of open-source LLMs designed for multi-scenario HDL generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06821v1">ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 This paper has been presented at IEEE VLSI Test Symposium (VTS) 2025
    </div>
    <details class="paper-abstract">
      Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07888v1">Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06782v1">Utilizing LLMs to Investigate the Disputed Role of Evidence in Electronic Cigarette Health Policy Formation in Australia and the UK</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Australia and the UK have developed contrasting approaches to the regulation of electronic cigarettes, with - broadly speaking - Australia adopting a relatively restrictive approach and the UK adopting a more permissive approach. Notably, these divergent policies were developed from the same broad evidence base. In this paper, to investigate differences in how the two jurisdictions manage and present evidence, we developed and evaluated a Large Language Model-based sentence classifier to perform automated analyses of electronic cigarette-related policy documents drawn from official Australian and UK legislative processes (109 documents in total). Specifically, we utilized GPT-4 to automatically classify sentences based on whether they contained claims that e-cigarettes were broadly helpful or harmful for public health. Our LLM-based classifier achieved an F-score of 0.9. Further, when applying the classifier to our entire sentence-level corpus, we found that Australian legislative documents show a much higher proportion of harmful statements, and a lower proportion of helpful statements compared to the expected values, with the opposite holding for the UK. In conclusion, this work utilized an LLM-based approach to provide evidence to support the contention that - drawing on the same evidence base - Australian ENDS-related policy documents emphasize the harms associated with ENDS products and UK policy documents emphasize the benefits. Further, our approach provides a starting point for using LLM-based methods to investigate the complex relationship between evidence and health policy formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19044v2">Calibrating Translation Decoding with Quality Estimation on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Neural machine translation (NMT) systems typically employ maximum a posteriori (MAP) decoding to select the highest-scoring translation from the distribution mass. However, recent evidence highlights the inadequacy of MAP decoding, often resulting in low-quality or even pathological hypotheses -- the decoding objective is not aligned with real-world translation quality. This paper proposes calibrating hypothesis likelihoods with translation quality from a distribution view by directly optimizing their Pearson correlation -- thereby enhancing the effectiveness of translation decoding. With our method, translation on large language models (LLMs) improves substantially after limited training (2K instances per direction). This improvement is orthogonal to those achieved through supervised fine-tuning, leading to substantial gains across a broad range of metrics and human evaluations -- even when applied to top-performing translation-specialized LLMs fine-tuned on high-quality translation data, such as Tower, or when compared to recent preference optimization methods, like CPO. Moreover, the calibrated translation likelihood can directly serve as a strong proxy for translation quality, closely approximating or even surpassing some state-of-the-art translation quality estimation models, like CometKiwi. Lastly, our in-depth analysis demonstrates that calibration enhances the effectiveness of MAP decoding, thereby enabling greater efficiency in real-world deployment. The resulting state-of-the-art translation model, which covers 10 languages, along with the accompanying code and human evaluation data, has been released to the community: https://github.com/moore3930/calibrating-llm-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v2">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06653v1">Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demand extensive memory capacity during both fine-tuning and inference. To enable memory-efficient fine-tuning, existing methods apply block-wise quantization techniques, such as NF4 and AF4, to the network weights. We show that these quantization techniques incur suboptimal quantization errors. Therefore, as a first novelty, we propose an optimization approach for block-wise quantization. Using this method, we design a family of quantizers named 4-bit block-wise optimal float (BOF4), which consistently reduces the quantization error compared to both baseline methods. We provide both a theoretical and a data-driven solution for the optimization process and prove their practical equivalence. Secondly, we propose a modification to the employed normalization method based on the signed absolute block maximum (BOF4-S), enabling further reduction of the quantization error and empirically achieving less degradation in language modeling performance. Thirdly, we explore additional variations of block-wise quantization methods applied to LLMs through an experimental study on the importance of accurately representing zero and large-amplitude weights on the one hand, and optimization towards various error metrics on the other hand. Lastly, we introduce a mixed-precision quantization strategy dubbed outlier-preserving quantization (OPQ) to address the distributional mismatch induced by outlier weights in block-wise quantization. By storing outlier weights in 16-bit precision (OPQ) while applying BOF4-S, we achieve top performance among 4-bit block-wise quantization techniques w.r.t. perplexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06591v1">Evaluating LLM-Generated Q&A Test: a Student-Centered Study</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 accepted to AIED 2025
    </div>
    <details class="paper-abstract">
      This research prepares an automatic pipeline for generating reliable question-answer (Q&A) tests using AI chatbots. We automatically generated a GPT-4o-mini-based Q&A test for a Natural Language Processing course and evaluated its psychometric and perceived-quality metrics with students and experts. A mixed-format IRT analysis showed that the generated items exhibit strong discrimination and appropriate difficulty, while student and expert star ratings reflect high overall quality. A uniform DIF check identified two items for review. These findings demonstrate that LLM-generated assessments can match human-authored tests in psychometric performance and user satisfaction, illustrating a scalable approach to AI-assisted assessment development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09798v2">Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06481v1">QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      The deployment of mixture-of-experts (MoE) large language models (LLMs) presents significant challenges due to their high memory demands. These challenges become even more pronounced in multi-tenant environments, where shared resources must accommodate multiple models, limiting the effectiveness of conventional virtualization techniques. This paper addresses the problem of efficiently serving multiple fine-tuned MoE-LLMs on a single-GPU. We propose a serving system that employs \textit{similarity-based expert consolidation} to reduce the overall memory footprint by sharing similar experts across models. To ensure output quality, we introduce \textit{runtime partial reconfiguration}, dynamically replacing non-expert layers when processing requests from different models. As a result, our approach achieves a competitive output quality while maintaining throughput comparable to serving a single model while incurring a negligible increase in time-to-first-token (TTFT). Experiments on a server with a single NVIDIA A100 GPU (80GB) using Mixtral-8x7B models demonstrate an 85\% average reduction in turnaround time compared to NVIDIA's multi-instance GPU (MIG). Furthermore, experiments on Google's Switch Transformer Base-8 model with up to four variants demonstrate the scalability and resilience of our approach in maintaining output quality compared to other model merging baselines, highlighting its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07877v1">Efficient Telecom Specific LLM: TSLAM-Mini with QLoRA and Digital Twin Data</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 Introducing TSLAM-Mini, a specialized language model for telecommunications, demonstrating the efficacy of QLoRA fine-tuning and digital twin-synthesized data for enhanced network intelligence. Model available on: https://huggingface.co/NetoAISolutions/TSLAM-Mini-2B
    </div>
    <details class="paper-abstract">
      General-purpose large language models (LLMs), despite their broad capabilities accrued from open-world data, frequently exhibit suboptimal performance when confronted with the nuanced and specialized demands inherent in real-time telecommunications applications. This investigation addresses this critical limitation through the meticulous fine-tuning of TSLAM-Mini developed by NetoAI, a compact (3.8-billion parameter) causal language model architecturally derived from Phi-4 Mini Instruct 4B. The fine-tuning regimen leverages a bespoke dataset comprising 100,000 samples, strategically engineered to address 20 pivotal telecommunications use-cases, encompassing domains such as Network Fundamentals, IP Routing, MPLS, Network Security, Automation, OSS/BSS, RAN, Mobile Core, Satellite Communications, and Ethical AI. This dataset was curated utilizing NetoAI's DigiTwin platform, enriched with granular insights from venerated network Subject Matter Experts (SMEs) and authoritative RFC documents, thereby capturing high-fidelity representations of real-world network dynamics through simulations inspired by digital twin paradigms. Employing Quantized Low-Rank Adaptation (QLoRA), a state-of-the-art Parameter Efficient Fine-Tuning (PEFT) technique, we achieved substantial training efficiency and enabled prospective deployment on resource-constrained hardware. A novel evaluation framework, predicated on a high-capacity LLM (Qwen3-235B-A22B) functioning as an automated adjudicator, was instituted to rigorously assess instruction-following fidelity and response quality across the specified telecom use-cases. Empirical results unequivocally demonstrate TSLAM-Mini's superior aptitude in telecom-centric applications, underscoring the profound efficacy of domain-specific datasets and PEFT methodologies for advancing intelligent network management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v2">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Update WIP
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LiTransProQA, a novel, reference-free, LLM-based question-answering framework designed specifically for literary translation evaluation. LiTransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LiTransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LiTransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04134v2">The Order Effect: Investigating Prompt Sensitivity to Input Order in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 The first 3 authors have contributed equally
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integral to diverse applications, ensuring their reliability under varying input conditions is crucial. One key issue affecting this reliability is order sensitivity, wherein slight variations in the input arrangement can lead to inconsistent or biased outputs. Although recent advances have reduced this sensitivity, the problem remains unresolved. This paper investigates the extent of order sensitivity in LLMs whose internal components are hidden from users (such as closed-source models or those accessed via API calls). We conduct experiments across multiple tasks, including paraphrasing, relevance judgment, and multiple-choice questions. Our results show that input order significantly affects performance across tasks, with shuffled inputs leading to measurable declines in output accuracy. Few-shot prompting demonstrates mixed effectiveness and offers partial mitigation; however, fails to fully resolve the problem. These findings highlight persistent risks, particularly in high-stakes applications, and point to the need for more robust LLMs or improved input-handling techniques in future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06184v1">From Millions of Tweets to Actionable Insights: Leveraging LLMs for User Profiling</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted at MisD @ AAAI ICWSM 2025
    </div>
    <details class="paper-abstract">
      Social media user profiling through content analysis is crucial for tasks like misinformation detection, engagement prediction, hate speech monitoring, and user behavior modeling. However, existing profiling techniques, including tweet summarization, attribute-based profiling, and latent representation learning, face significant limitations: they often lack transferability, produce non-interpretable features, require large labeled datasets, or rely on rigid predefined categories that limit adaptability. We introduce a novel large language model (LLM)-based approach that leverages domain-defining statements, which serve as key characteristics outlining the important pillars of a domain as foundations for profiling. Our two-stage method first employs semi-supervised filtering with a domain-specific knowledge base, then generates both abstractive (synthesized descriptions) and extractive (representative tweet selections) user profiles. By harnessing LLMs' inherent knowledge with minimal human validation, our approach is adaptable across domains while reducing the need for large labeled datasets. Our method generates interpretable natural language user profiles, condensing extensive user data into a scale that unlocks LLMs' reasoning and knowledge capabilities for downstream social network tasks. We contribute a Persian political Twitter (X) dataset and an LLM-based evaluation framework with human validation. Experimental results show our method significantly outperforms state-of-the-art LLM-based and traditional methods by 9.8%, demonstrating its effectiveness in creating flexible, adaptable, and interpretable user profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06150v1">A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06149v1">Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Despite growing interest in automated hate speech detection, most existing approaches overlook the linguistic diversity of online content. Multilingual instruction-tuned large language models such as LLaMA, Aya, Qwen, and BloomZ offer promising capabilities across languages, but their effectiveness in identifying hate speech through zero-shot and few-shot prompting remains underexplored. This work evaluates LLM prompting-based detection across eight non-English languages, utilizing several prompting techniques and comparing them to fine-tuned encoder models. We show that while zero-shot and few-shot prompting lag behind fine-tuned encoder models on most of the real-world evaluation sets, they achieve better generalization on functional tests for hate speech detection. Our study also reveals that prompt design plays a critical role, with each language often requiring customized prompting techniques to maximize performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09667v2">k-LLMmeans: Scalable, Stable, and Interpretable Text Clustering via LLM-based Centroids</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      We introduce k-LLMmeans, a novel modification of the k-means algorithm for text clustering that leverages LLM-generated summaries as cluster centroids, capturing semantic nuances often missed by purely numerical averages. This design preserves the core optimization properties of k-means while enhancing semantic interpretability and avoiding the scalability and instability issues typical of modern LLM-based clustering. Unlike existing methods, our approach does not increase LLM usage with dataset size and produces transparent intermediate outputs. We further extend it with a mini-batch variant for efficient, real-time clustering of streaming text. Extensive experiments across multiple datasets, embeddings, and LLMs show that k-LLMmeans consistently outperforms k-means and other traditional baselines and achieves results comparable to state-of-the-art LLM-based clustering, with a fraction of the LLM calls. Finally, we present a case study on sequential text streams and introduce a new benchmark dataset constructed from StackExchange to evaluate text-stream clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06120v1">LLMs Get Lost In Multi-Turn Conversation</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are conversational interfaces. As such, LLMs have the potential to assist their users not only when they can fully specify the task at hand, but also to help them define, explore, and refine what they need through multi-turn conversational exchange. Although analysis of LLM conversation logs has confirmed that underspecification occurs frequently in user instructions, LLM evaluation has predominantly focused on the single-turn, fully-specified instruction setting. In this work, we perform large-scale simulation experiments to compare LLM performance in single- and multi-turn settings. Our experiments confirm that all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely. In simpler terms, we discover that *when LLMs take a wrong turn in a conversation, they get lost and do not recover*.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06096v1">Free and Fair Hardware: A Pathway to Copyright Infringement-Free Verilog Generation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted at DAC 2025
    </div>
    <details class="paper-abstract">
      Limitations in Large Language Model (LLM) capabilities for hardware design tasks, such as generating functional Verilog codes, have motivated various fine-tuning optimizations utilizing curated hardware datasets from open-source repositories. However, these datasets remain limited in size and contain minimal checks on licensing for reuse, resulting in potential copyright violations by fine-tuned LLMs. Therefore, we propose an evaluation benchmark to estimate the risk of Verilog-trained LLMs to generate copyright-protected codes. To minimize this risk, we present an open-source Verilog dataset, FreeSet, containing over 220k files, along with the automated dataset curation framework utilized to provide additional guarantees of fair-use Verilog data. We then execute an LLM fine-tuning framework consisting of continual pre-training, resulting in a fine-tuned Llama model for Verilog, FreeV. Our results indicate that FreeV demonstrates the smallest risk of copyright-infringement among prior works, with only a 3% violation rate. Furthermore, experimental results demonstrate improvements in Verilog generation functionality over its baseline model, improving VerilogEval pass@10 rates by over 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06046v1">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 24 pages, 10 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries, created via an automated pipeline. We also release a new dataset of the extracted UK Government public health guidance documents used as source text for PubHealthBench. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, whilst there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses on public health topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11963v2">NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Context?</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 v2: updated with tested models and Multi-Needle Reasoning implementation
    </div>
    <details class="paper-abstract">
      The capability of large language models to handle long-context information is crucial across various real-world applications. Existing evaluation methods often rely either on real-world long texts, making it difficult to exclude the influence of models' inherent knowledge, or introduce irrelevant filler content to artificially achieve target lengths, reducing assessment effectiveness. To address these limitations, we introduce NeedleBench, a synthetic framework for assessing retrieval and reasoning performance in bilingual long-context tasks with adaptive context lengths. NeedleBench systematically embeds key data points at varying depths to rigorously test model capabilities. Tasks are categorized into two scenarios: information-sparse, featuring minimal relevant details within extensive irrelevant text to simulate simple retrieval tasks; and information-dense (the Ancestral Trace Challenge), where relevant information is continuously distributed throughout the context to simulate complex reasoning tasks. Our experiments reveal that although recent reasoning models like Deepseek-R1 and OpenAI's o3 excel in mathematical reasoning, they struggle with continuous retrieval and reasoning in information-dense scenarios, even at shorter context lengths. We also characterize a phenomenon termed 'under-thinking', where models prematurely conclude reasoning despite available information. NeedleBench thus provides critical insights and targeted tools essential for evaluating and improving LLMs' long-context capabilities. All resources are available at OpenCompass: https://github.com/open-compass/opencompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05832v1">Augmented Body Communicator: Enhancing daily body expression for people with upper limb limitations through LLM and a robotic arm</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Individuals with upper limb movement limitations face challenges in interacting with others. Although robotic arms are currently used primarily for functional tasks, there is considerable potential to explore ways to enhance users' body language capabilities during social interactions. This paper introduces an Augmented Body Communicator system that integrates robotic arms and a large language model. Through the incorporation of kinetic memory, disabled users and their supporters can collaboratively design actions for the robot arm. The LLM system then provides suggestions on the most suitable action based on contextual cues during interactions. The system underwent thorough user testing with six participants who have conditions affecting upper limb mobility. Results indicate that the system improves users' ability to express themselves. Based on our findings, we offer recommendations for developing robotic arms that support disabled individuals with body language capabilities and functional tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v5">Estimating LLM Uncertainty with Evidence</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Over the past few years, Large Language Models (LLMs) have developed rapidly and are widely applied in various domains. However, LLMs face the issue of hallucinations, generating responses that may be unreliable when the models lack relevant knowledge. To be aware of potential hallucinations, uncertainty estimation methods have been introduced, and most of them have confirmed that reliability lies in critical tokens. However, probability-based methods perform poorly in identifying token reliability, limiting their practical utility. In this paper, we reveal that the probability-based method fails to estimate token reliability due to the loss of evidence strength information which is accumulated in the training stage. Therefore, we present Logits-induced token uncertainty (LogTokU), a framework for estimating decoupled token uncertainty in LLMs, enabling real-time uncertainty estimation without requiring multiple sampling processes. We employ evidence modeling to implement LogTokU and use the estimated uncertainty to guide downstream tasks. The experimental results demonstrate that LogTokU has significant effectiveness and promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05794v1">What Is Next for LLMs? Next-Generation AI Computing Hardware Using Photonic Chips</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 36 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly pushing the limits of contemporary computing hardware. For example, training GPT-3 has been estimated to consume around 1300 MWh of electricity, and projections suggest future models may require city-scale (gigawatt) power budgets. These demands motivate exploration of computing paradigms beyond conventional von Neumann architectures. This review surveys emerging photonic hardware optimized for next-generation generative AI computing. We discuss integrated photonic neural network architectures (e.g., Mach-Zehnder interferometer meshes, lasers, wavelength-multiplexed microring resonators) that perform ultrafast matrix operations. We also examine promising alternative neuromorphic devices, including spiking neural network circuits and hybrid spintronic-photonic synapses, which combine memory and processing. The integration of two-dimensional materials (graphene, TMDCs) into silicon photonic platforms is reviewed for tunable modulators and on-chip synaptic elements. Transformer-based LLM architectures (self-attention and feed-forward layers) are analyzed in this context, identifying strategies and challenges for mapping dynamic matrix multiplications onto these novel hardware substrates. We then dissect the mechanisms of mainstream LLMs, such as ChatGPT, DeepSeek, and LLaMA, highlighting their architectural similarities and differences. We synthesize state-of-the-art components, algorithms, and integration methods, highlighting key advances and open issues in scaling such systems to mega-sized LLM models. We find that photonic computing systems could potentially surpass electronic processors by orders of magnitude in throughput and energy efficiency, but require breakthroughs in memory, especially for long-context windows and long token sequences, and in storage of ultra-large datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05786v1">A Day in Their Shoes: Using LLM-Based Perspective-Taking Interactive Fiction to Reduce Stigma Toward Dirty Work</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Conference paper for FAccT '25
    </div>
    <details class="paper-abstract">
      Occupations referred to as "dirty work" often face entrenched social stigma, which adversely affects the mental health of workers in these fields and impedes occupational equity. In this study, we propose a novel Interactive Fiction (IF) framework powered by Large Language Models (LLMs) to encourage perspective-taking and reduce biases against these stigmatized yet essential roles. Through an experiment with participants (n = 100) across four such occupations, we observed a significant increase in participants' understanding of these occupations, as well as a high level of empathy and a strong sense of connection to individuals in these roles. Additionally, qualitative interviews with participants (n = 15) revealed that the LLM-based perspective-taking IF enhanced immersion, deepened emotional resonance and empathy toward "dirty work," and allowed participants to experience a sense of professional fulfillment in these occupations. However, participants also highlighted ongoing challenges, such as limited contextual details generated by the LLM and the unintentional reinforcement of existing stereotypes. Overall, our findings underscore that an LLM-based perspective-taking IF framework offers a promising and scalable strategy for mitigating stigma and promoting social equity in marginalized professions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21223v3">Rethinking Graph Structure Learning in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 29 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recently, the emergence of LLMs has prompted researchers to integrate language descriptions into graphs, aiming to enhance model encoding capabilities from a data-centric perspective. This graph representation is called text-attributed graphs (TAGs). A review of prior advancements highlights that graph structure learning (GSL) is a pivotal technique for improving data utility, making it highly relevant to efficient TAG learning. However, most GSL methods are tailored for traditional graphs without textual information, underscoring the necessity of developing a new GSL paradigm. Despite clear motivations, it remains challenging: (1) How can we define a reasonable optimization objective for GSL in the era of LLMs, considering the massive parameters in LLM? (2) How can we design an efficient model architecture that enables seamless integration of LLM for this optimization objective? For Question 1, we reformulate existing GSL optimization objectives as a tree optimization framework, shifting the focus from obtaining a well-trained edge predictor to a language-aware tree sampler. For Question 2, we propose decoupled and training-free model design principles for LLM integration, shifting the focus from computation-intensive fine-tuning to more efficient inference. Based on this, we propose Large Language and Tree Assistant (LLaTA), which leverages tree-based LLM in-context learning to enhance the understanding of topology and text, enabling reliable inference and generating improved graph structure. Extensive experiments on 10 datasets demonstrate that LLaTA enjoys flexibility-incorporated with any backbone; scalability-outperforms other LLM-enhanced graph learning methods; effectiveness-achieves SOTA predictive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05772v1">Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Transformer-based models are the foundation of modern machine learning, but their execution, particularly during autoregressive decoding in large language models (LLMs), places significant pressure on memory systems due to frequent memory accesses and growing key-value (KV) caches. This creates a bottleneck in memory bandwidth, especially as context lengths increase. Processing-in-memory (PIM) architectures are a promising solution, offering high internal bandwidth and compute parallelism near memory. However, current PIM designs are primarily optimized for dense attention and struggle with the dynamic, irregular access patterns introduced by modern KV cache sparsity techniques. Consequently, they suffer from workload imbalance, reducing throughput and resource utilization. In this work, we propose STARC, a novel sparsity-optimized data mapping scheme tailored specifically for efficient LLM decoding on PIM architectures. STARC clusters KV pairs by semantic similarity and maps them to contiguous memory regions aligned with PIM bank structures. During decoding, queries retrieve relevant tokens at cluster granularity by matching against precomputed centroids, enabling selective attention and parallel processing without frequent reclustering or data movement overhead. Experiments on the HBM-PIM system show that, compared to common token-wise sparsity methods, STARC reduces attention-layer latency by 19%--31% and energy consumption by 19%--27%. Under a KV cache budget of 1024, it achieves up to 54%--74% latency reduction and 45%--67% energy reduction compared to full KV cache retrieval. Meanwhile, STARC maintains model accuracy comparable to state-of-the-art sparse attention methods, demonstrating its effectiveness in enabling efficient and hardware-friendly long-context LLM inference on PIM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05762v1">Multi-Agent Systems for Robotic Autonomy with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 11 pages, 2 figures, 5 tables, submitted for publication
    </div>
    <details class="paper-abstract">
      Since the advent of Large Language Models (LLMs), various research based on such models have maintained significant academic attention and impact, especially in AI and robotics. In this paper, we propose a multi-agent framework with LLMs to construct an integrated system for robotic task analysis, mechanical design, and path generation. The framework includes three core agents: Task Analyst, Robot Designer, and Reinforcement Learning Designer. Outputs are formatted as multimodal results, such as code files or technical reports, for stronger understandability and usability. To evaluate generalizability comparatively, we conducted experiments with models from both GPT and DeepSeek. Results demonstrate that the proposed system can design feasible robots with control strategies when appropriate task inputs are provided, exhibiting substantial potential for enhancing the efficiency and accessibility of robotic system development in research and industrial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v1">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05744v1">Harnessing LLMs Explanations to Boost Surrogate Models in Tabular Data Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable ability in solving complex tasks, making them a promising tool for enhancing tabular learning. However, existing LLM-based methods suffer from high resource requirements, suboptimal demonstration selection, and limited interpretability, which largely hinder their prediction performance and application in the real world. To overcome these problems, we propose a novel in-context learning framework for tabular prediction. The core idea is to leverage the explanations generated by LLMs to guide a smaller, locally deployable Surrogate Language Model (SLM) to make interpretable tabular predictions. Specifically, our framework mainly involves three stages: (i) Post Hoc Explanation Generation, where LLMs are utilized to generate explanations for question-answer pairs in candidate demonstrations, providing insights into the reasoning behind the answer. (ii) Post Hoc Explanation-Guided Demonstrations Selection, which utilizes explanations generated by LLMs to guide the process of demonstration selection from candidate demonstrations. (iii) Post Hoc Explanation-Guided Interpretable SLM Prediction, which utilizes the demonstrations obtained in step (ii) as in-context and merges corresponding explanations as rationales to improve the performance of SLM and guide the model to generate interpretable outputs. Experimental results highlight the framework's effectiveness, with an average accuracy improvement of 5.31% across various tabular datasets in diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05712v1">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text, either to the specific language model used or to the individual user who generated it. This is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this issue is watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, which enables the recovery of a secret author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points (x, f(x)) that lie on a single straight line. The x-coordinates are generated pseudorandomly using either an LFSR (when security is not a priority) or a cryptographically secure NFSR for high-security applications. The scheme efficiency and resilience to adversarial modifications are analysed. Experimental results show that the proposed method is highly effective, allowing the recovery of the author identity when as few as three points survive adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v3">Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) handle increasingly longer contexts, serving long inference requests of millions of tokens presents unique challenges. We show that existing work for long context inference is largely based on techniques from long context training, and does not handle the high variability in input lengths during inference. This leads to inefficient resource utilization, server fragmentation, and head-of-line (HOL) blocking. We present Medha, an end-to-end system for efficient long-context LLM inference that addresses these challenges through fine-grained time sharing. Medha introduces three key innovations: (1) the mechanism of adaptive prefill chunking to help mitigate HOL blocking with preemption; (2) two new parallelism strategies: Sequence Pipeline Parallelism (SPP) to reduce time-to-first-token by pipelining prefill chunks, and KV-Cache Parallelism (KVP) to lower time-peroutput-token by distributing decoding across servers; and (3) a novel input-length aware least remaining slack scheduling to meet Service Level Objectives (SLOs). Medha enables exact inference scaling beyond 10 million tokens, maintaining high throughput and low latency across mixed-length workloads. Compared to state-of-the-art systems, Medha reduces server fragmentation, cuts median latency by up to 30x, and improves throughput by over 5x, delivering production-scale long-context inference without compromising performance on shorter requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v3">Token-Efficient RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Title updated to "Token-Efficient RL for LLM Reasoning" to better reflect algorithmic focus. Revised abstract, intro, and conclusion. Paper shortened and typos fixed
    </div>
    <details class="paper-abstract">
      We propose reinforcement learning (RL) strategies tailored for reasoning in large language models (LLMs) under strict memory and compute limits, with a particular focus on compatibility with LoRA fine-tuning. Building on early policy gradient methods with baseline subtraction, we design critic-free methods that operate on a small, informative subset of output tokens to reduce memory usage and stabilize training. We introduce S-GRPO, a stochastic variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching approach for fine-grained credit assignment. Applied to Qwen2-1.5B, our methods raise accuracy on the SVAMP benchmark from 46% to over 70% and show strong performance on multi-digit multiplication. Surprisingly, full-token GRPO under LoRA fails to improve over the base model, suggesting that selective token-level optimization may act as an implicit regularizer in low-parameter training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06469v1">KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted to the Educational Data Mining (EDM) 2025 conference
    </div>
    <details class="paper-abstract">
      Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18966v3">Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 This paper is accepted by NAACL 2025 findings. Link to the paper presentation: https://youtu.be/IhaxwbZOcaU
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06461v1">Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      The common assumption in on-device AI is that GPUs, with their superior parallel processing, always provide the best performance for large language model (LLM) inference. In this work, we challenge this notion by empirically demonstrating that, under certain conditions, CPUs can outperform GPUs for LLM inference on mobile devices. Using a 1-billion-parameter LLM deployed via llama.cpp on the iPhone 15 Pro, we show that a CPU-only configuration (two threads, F16 precision) achieves 17 tokens per second, surpassing the 12.8 tokens per second obtained with GPU acceleration. We analyze the architectural factors driving this counterintuitive result, revealing that GPU memory transfer overhead and CPU thread optimization play a critical role. Furthermore, we explore the impact of thread oversubscription, quantization strategies, and hardware constraints, providing new insights into efficient on-device AI execution. Our findings challenge conventional GPU-first thinking, highlighting the untapped potential of optimized CPU inference and paving the way for smarter deployment strategies in mobile AI. However, fully explaining the observed CPU advantage remains difficult due to limited access to low-level profiling tools on iOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06438v1">Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06416v1">ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and the introduction of the Model Context Protocol (MCP) have significantly expanded LLM agents' capability to interact dynamically with external tools and APIs. However, existing tool selection frameworks do not integrate MCP servers, instead relying heavily on error-prone manual updates to monolithic local tool repositories, leading to duplication, inconsistencies, and inefficiencies. Additionally, current approaches abstract tool selection before the LLM agent is invoked, limiting its autonomy and hindering dynamic re-querying capabilities during multi-turn interactions. To address these issues, we introduce ScaleMCP, a novel tool selection approach that dynamically equips LLM agents with a MCP tool retriever, giving agents the autonomy to add tools into their memory, as well as an auto-synchronizing tool storage system pipeline through CRUD (create, read, update, delete) operations with MCP servers as the single source of truth. We also propose a novel embedding strategy, Tool Document Weighted Average (TDWA), designed to selectively emphasize critical components of tool documents (e.g. tool name or synthetic questions) during the embedding process. Comprehensive evaluations conducted on a created dataset of 5,000 financial metric MCP servers, across 10 LLM models, 5 embedding models, and 5 retriever types, demonstrate substantial improvements in tool retrieval and agent invocation performance, emphasizing ScaleMCP's effectiveness in scalable, dynamic tool selection and invocation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06003v2">LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Conference Version (ISCA'25)
    </div>
    <details class="paper-abstract">
      As large language model (LLM) inference continues to demand increasing computational resources, there is a rapidly growing trend toward using low-bit weights to reduce memory footprint and improve inference efficiency. However, low-bit LLMs introduce the need for mixed-precision general matrix multiplication (mpGEMM), which involves multiplying low-precision weights with higher-precision activations - a critical yet under-explored operation. Current hardware lacks native support for mpGEMM, leading to inefficient dequantization-based implementations. To address this, we explore a lookup table (LUT)-based approach to accelerate mpGEMM. While conventional LUT implementations fall short in performance and flexibility, we propose LUT Tensor Core, a software-hardware co-designed solution optimized for low-bit LLM inference. On the software side, we introduce operator fusion and table symmetrization techniques to optimize LUT generation and storage. On the hardware side, LUT Tensor Core adopts an elongated tiling shape to maximize table reuse and employs a bit-serial architecture to flexibly support a variety of precision combinations. Additionally, we design an end-to-end compilation stack with custom instructions to enable efficient code generation and optimization for LUT-based mpGEMM. Experimental results on low-bit LLMs such as BitNet and LLaMA demonstrate that LUT Tensor Core delivers over an order-of-magnitude improvement in both compute density and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09701v2">Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted and to appear in IJCNN 2025
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for program code generation has gained substantial attention, but their biases and limitations with non-English prompts challenge global inclusivity. This paper investigates the complexities of multilingual prompt-based code generation. Our evaluations of LLMs, including CODELLAMA and CODEGEMMA, reveal significant disparities in code quality for non-English prompts; we also demonstrate the inadequacy of simple approaches like prompt translation, bootstrapped data augmentation, and fine-tuning. To address this, we propose a zero-shot cross-lingual approach using a neural projection technique, integrating a cross-lingual encoder like LASER to map multilingual embeddings from it into the LLM's token space. This method requires training only on English data and scales effectively to other languages. Results on a translated and quality-checked MBPP dataset show substantial improvements in code quality. This research promotes a more inclusive code generation landscape by empowering LLMs with multilingual capabilities to support the diverse linguistic spectrum in programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06364v1">LATENT: LLM-Augmented Trojan Insertion and Evaluation Framework for Analog Netlist Topologies</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted for presentation at IEEE International Conference on LLM-Aided Design (ICLAD), 2025
    </div>
    <details class="paper-abstract">
      Analog and mixed-signal (A/MS) integrated circuits (ICs) are integral to safety-critical applications. However, the globalization and outsourcing of A/MS ICs to untrusted third-party foundries expose them to security threats, particularly analog Trojans. Unlike digital Trojans which have been extensively studied, analog Trojans remain largely unexplored. There has been only limited research on their diversity and stealth in analog designs, where a Trojan is activated only during a narrow input voltage range. Effective defense techniques require a clear understanding of the attack vectors; however, the lack of diverse analog Trojan instances limits robust advances in detection strategies. To address this gap, we present LATENT, the first large language model (LLM)-driven framework for crafting stealthy, circuit-specific analog Trojans. LATENT incorporates LLM as an autonomous agent to intelligently insert and refine Trojan components within analog designs based on iterative feedback from a detection model. This feedback loop ensures that the inserted Trojans remain stealthy while successfully evading detection. Experimental results demonstrate that our generated Trojan designs exhibit an average Trojan-activation range of 15.74%, ensuring they remain inactive under most operating voltages, while causing a significant performance degradation of 11.3% upon activation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06321v1">Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in https://github.com/zch65458525/L2T.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18635v2">Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      While Retrieval Augmented Generation (RAG) has emerged as a popular technique for improving Large Language Model (LLM) systems, it introduces a large number of choices, parameters and hyperparameters that must be made or tuned. This includes the LLM, embedding, and ranker models themselves, as well as hyperparameters governing individual RAG components. Yet, collectively optimizing the entire configuration in a RAG or LLM system remains under-explored - especially in multi-objective settings - due to intractably large solution spaces, noisy objective evaluations, and the high cost of evaluations. In this work, we introduce the first approach for multi-objective parameter optimization of cost, latency, safety and alignment over entire LLM and RAG systems. We find that Bayesian optimization methods significantly outperform baseline approaches, obtaining a superior Pareto front on two new RAG benchmark tasks. We conclude our work with important considerations for practitioners who are designing multi-objective RAG systems, highlighting nuances such as how optimal configurations may not generalize across tasks and objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05057v1">Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted by FSE 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Application Programming Interfaces (APIs) are crucial in modern software development. Large Language Models (LLMs) assist in automated code generation but often struggle with API hallucination, including invoking non-existent APIs and misusing existing ones in practical development scenarios. Existing studies resort to Retrieval-Augmented Generation (RAG) methods for mitigating the hallucination issue, but tend to fail since they generally ignore the structural dependencies in practical projects and do not indeed validate whether the generated APIs are available or not. To address these limitations, we propose MARIN, a framework for mitigating API hallucination in code generated by LLMs with hierarchical dependency aware. MARIN consists of two phases: Hierarchical Dependency Mining, which analyzes local and global dependencies of the current function, aiming to supplement comprehensive project context in LLMs input, and Dependency Constrained Decoding, which utilizes mined dependencies to adaptively constrain the generation process, aiming to ensure the generated APIs align with the projects specifications. To facilitate the evaluation of the degree of API hallucination, we introduce a new benchmark APIHulBench and two new metrics including Micro Hallucination Number (MiHN) and Macro Hallucination Rate (MaHR). Experiments on six state-of-the-art LLMs demonstrate that MARIN effectively reduces API hallucinations, achieving an average decrease of 67.52% in MiHN and 73.56% in MaHR compared to the RAG approach. Applied to Huaweis internal projects and two proprietary LLMs, MARIN achieves average decreases of 57.33% in MiHN and 59.41% in MaHR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v2">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4% absolute on SAT MATH). Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03961v2">The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 16 pages, 8 figures. Code available at https://github.com/storyagents25/story-agents
    </div>
    <details class="paper-abstract">
      According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00762v4">Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scales Test-Time Compute</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05016v1">The Pitfalls of Growing Group Complexity: LLMs and Social Choice-Based Aggregation for Group Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 To be published in: Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization (UMAP Adjunct '25), June 16--19, 2025, New York City, NY, USA Accepted at the 4th Workshop on Group Modeling, Adaptation and Personalization (GMAP), co-located at UMAP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied in recommender systems aimed at both individuals and groups. Previously, Group Recommender Systems (GRS) often used social choice-based aggregation strategies to derive a single recommendation based on the preferences of multiple people. In this paper, we investigate under which conditions language models can perform these strategies correctly based on zero-shot learning and analyse whether the formatting of the group scenario in the prompt affects accuracy. We specifically focused on the impact of group complexity (number of users and items), different LLMs, different prompting conditions, including In-Context learning or generating explanations, and the formatting of group preferences. Our results show that performance starts to deteriorate when considering more than 100 ratings. However, not all language models were equally sensitive to growing group complexity. Additionally, we showed that In-Context Learning (ICL) can significantly increase the performance at higher degrees of group complexity, while adding other prompt modifications, specifying domain cues or prompting for explanations, did not impact accuracy. We conclude that future research should include group complexity as a factor in GRS evaluation due to its effect on LLM performance. Furthermore, we showed that formatting the group scenarios differently, such as rating lists per user or per item, affected accuracy. All in all, our study implies that smaller LLMs are capable of generating group recommendations under the right conditions, making the case for using smaller models that require less computing power and costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14401v2">LLM-Driven Usefulness Judgment for Web Search Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06877v3">LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 11 pages. Accepted at SIGIR 2025 (48th International ACM SIGIR Conference on Research and Development in Information Retrieval)
    </div>
    <details class="paper-abstract">
      Test collections are information retrieval tools that allow researchers to quickly and easily evaluate ranking algorithms. While test collections have become an integral part of IR research, the process of data creation involves significant effort in manual annotations, which often makes it very expensive and time-consuming. Thus, test collections could become too small when the budget is limited, which may lead to unstable evaluations. As a cheaper alternative, recent studies have proposed the use of large language models (LLMs) to completely replace human assessors. However, while LLMs seem to somewhat correlate with human judgments, their predictions are not perfect and often show bias. Thus a complete replacement with LLMs is argued to be too risky and not fully reliable. Thus, in this paper, we propose LLM-Assisted Relevance Assessments (LARA), an effective method to balance manual annotations with LLM annotations, which helps to build a rich and reliable test collection even under a low budget. We use the LLM's predicted relevance probabilities to select the most profitable documents to manually annotate under a budget constraint. With theoretical reasoning, LARA effectively guides the human annotation process by actively learning to calibrate the LLM's predicted relevance probabilities. Then, using the calibration model learned from the limited manual annotations, LARA debiases the LLM predictions to annotate the remaining non-assessed data. Empirical evaluations on TREC-7 Ad Hoc, TREC-8 Ad Hoc, TREC Robust 2004, and TREC-COVID datasets show that LARA outperforms alternative solutions under almost any budget constraint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01259v2">Facilitating Instructors-LLM Collaboration for Problem Design in Introductory Programming Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs), such as ChatGPT, offer significant opportunities to enhance instructional support in introductory programming courses. While extensive research has explored the effectiveness of LLMs in supporting student learning, limited studies have examined how these models can assist instructors in designing instructional activities. This work investigates how instructors' expertise in effective activity design can be integrated with LLMs' ability to generate novel and targeted programming problems, facilitating more effective activity creation for programming classrooms. To achieve this, we employ a participatory design approach to develop an instructor-authoring tool that incorporates LLM support, fostering collaboration between instructors and AI in generating programming exercises. This tool also allows instructors to specify common student mistakes and misconceptions, which informs the adaptive feedback generation process. We conduct case studies with three instructors, analyzing how they use our system to design programming problems for their introductory courses. Through these case studies, we assess instructors' perceptions of the usefulness and limitations of LLMs in authoring problem statements for instructional purposes. Additionally, we compare the efficiency, quality, effectiveness, and coverage of designed activities when instructors create problems with and without structured LLM prompting guidelines. Our findings provide insights into the potential of LLMs in enhancing instructor workflows and improving programming education and provide guidelines for designing effective AI-assisted problem-authoring interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09798v3">ReadMe.LLM: A Framework to Help LLMs Understand Your Library</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 15 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with code generation tasks involving niche software libraries. Existing code generation techniques with only human-oriented documentation can fail -- even when the LLM has access to web search and the library is documented online. To address this challenge, we propose ReadMe$.$LLM, LLM-oriented documentation for software libraries. By attaching the contents of ReadMe$.$LLM to a query, performance consistently improves to near-perfect accuracy, with one case study demonstrating up to 100% success across all tested models. We propose a software development lifecycle where LLM-specific documentation is maintained alongside traditional software updates. In this study, we present two practical applications of the ReadMe$.$LLM idea with diverse software libraries, highlighting that our proposed approach could generalize across programming domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02309v2">Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted to IEEE COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02665v2">A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04948v1">Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Recommender systems are essential for delivering personalized content across digital platforms by modeling user preferences and behaviors. Recently, large language models (LLMs) have been adopted for prompt-based recommendation due to their ability to generate personalized outputs without task-specific training. However, LLM-based methods face limitations such as limited context window size, inefficient pointwise and pairwise prompting, and difficulty handling listwise ranking due to token constraints. LLMs can also be sensitive to position bias, as they may overemphasize earlier items in the prompt regardless of their true relevance. To address and investigate these issues, we propose a hybrid framework that combines a traditional recommendation model with an LLM for reranking top-k items using structured prompts. We evaluate the effects of user history reordering and instructional prompts for mitigating position bias. Experiments on MovieLens-100K show that randomizing user history improves ranking quality, but LLM-based reranking does not outperform the base model. Explicit instructions to reduce position bias are also ineffective. Our evaluations reveal limitations in LLMs' ability to model ranking context and mitigate bias. Our code is publicly available at https://github.com/aminul7506/LLMForReRanking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05665v1">Adaptive Stress Testing Black-Box LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 26 pages, 16 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing black-box methods often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using Adaptive Stress Testing (AST) with Monte-Carlo Tree Search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty. By generating MCTS prompt perturbation trees across diverse scenarios, we show that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05660v1">Not Like Us, Hunty: Measuring Perceptions and Behavioral Effects of Minoritized Anthropomorphic Cues in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted to FAccT 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly adapt and personalize to diverse sets of users, there is an increased risk of systems appropriating sociolects, i.e., language styles or dialects that are associated with specific minoritized lived experiences (e.g., African American English, Queer slang). In this work, we examine whether sociolect usage by an LLM agent affects user reliance on its outputs and user perception (satisfaction, frustration, trust, and social presence). We designed and conducted user studies where 498 African American English (AAE) speakers and 487 Queer slang speakers performed a set of question-answering tasks with LLM-based suggestions in either standard American English (SAE) or their self-identified sociolect. Our findings showed that sociolect usage by LLMs influenced both reliance and perceptions, though in some surprising ways. Results suggest that both AAE and Queer slang speakers relied more on the SAE agent, and had more positive perceptions of the SAE agent. Yet, only Queer slang speakers felt more social presence from the Queer slang agent over the SAE one, whereas only AAE speakers preferred and trusted the SAE agent over the AAE one. These findings emphasize the need to test for behavioral outcomes rather than simply assume that personalization would lead to a better and safer reliance outcome. They also highlight the nuanced dynamics of minoritized language in machine interactions, underscoring the need for LLMs to be carefully designed to respect cultural and linguistic boundaries while fostering genuine user engagement and trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05584v1">PRIMG : Efficient LLM-driven Test Generation Using Mutant Prioritization</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Mutation testing is a widely recognized technique for assessing and enhancing the effectiveness of software test suites by introducing deliberate code mutations. However, its application often results in overly large test suites, as developers generate numerous tests to kill specific mutants, increasing computational overhead. This paper introduces PRIMG (Prioritization and Refinement Integrated Mutation-driven Generation), a novel framework for incremental and adaptive test case generation for Solidity smart contracts. PRIMG integrates two core components: a mutation prioritization module, which employs a machine learning model trained on mutant subsumption graphs to predict the usefulness of surviving mutants, and a test case generation module, which utilizes Large Language Models (LLMs) to generate and iteratively refine test cases to achieve syntactic and behavioral correctness. We evaluated PRIMG on real-world Solidity projects from Code4Arena to assess its effectiveness in improving mutation scores and generating high-quality test cases. The experimental results demonstrate that PRIMG significantly reduces test suite size while maintaining high mutation coverage. The prioritization module consistently outperformed random mutant selection, enabling the generation of high-impact tests with reduced computational effort. Furthermore, the refining process enhanced the correctness and utility of LLM-generated tests, addressing their inherent limitations in handling edge cases and complex program logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05583v1">KG-HTC: Integrating Knowledge Graphs into LLMs for Effective Zero-shot Hierarchical Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Hierarchical Text Classification (HTC) involves assigning documents to labels organized within a taxonomy. Most previous research on HTC has focused on supervised methods. However, in real-world scenarios, employing supervised HTC can be challenging due to a lack of annotated data. Moreover, HTC often faces issues with large label spaces and long-tail distributions. In this work, we present Knowledge Graphs for zero-shot Hierarchical Text Classification (KG-HTC), which aims to address these challenges of HTC in applications by integrating knowledge graphs with Large Language Models (LLMs) to provide structured semantic context during classification. Our method retrieves relevant subgraphs from knowledge graphs related to the input text using a Retrieval-Augmented Generation (RAG) approach. Our KG-HTC can enhance LLMs to understand label semantics at various hierarchy levels. We evaluate KG-HTC on three open-source HTC datasets: WoS, DBpedia, and Amazon. Our experimental results show that KG-HTC significantly outperforms three baselines in the strict zero-shot setting, particularly achieving substantial improvements at deeper levels of the hierarchy. This evaluation demonstrates the effectiveness of incorporating structured knowledge into LLMs to address HTC's challenges in large label spaces and long-tailed label distributions. Our code is available at: https://github.com/QianboZang/KG-HTC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07861v1">Scalable LLM Math Reasoning Acceleration with Low-rank Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a low-cost distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>11% reduction to generate 2048 tokens with Qwen 2.5 14B) while encouraging response brevity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05445v1">clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05441v1">GesPrompt: Leveraging Co-Speech Gestures to Augment LLM-Based Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based copilots have shown great potential in Extended Reality (XR) applications. However, the user faces challenges when describing the 3D environments to the copilots due to the complexity of conveying spatial-temporal information through text or speech alone. To address this, we introduce GesPrompt, a multimodal XR interface that combines co-speech gestures with speech, allowing end-users to communicate more naturally and accurately with LLM-based copilots in XR environments. By incorporating gestures, GesPrompt extracts spatial-temporal reference from co-speech gestures, reducing the need for precise textual prompts and minimizing cognitive load for end-users. Our contributions include (1) a workflow to integrate gesture and speech input in the XR environment, (2) a prototype VR system that implements the workflow, and (3) a user study demonstrating its effectiveness in improving user communication in VR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05427v1">Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 The datasets are available on https://huggingface.co/datasets/openbmb/UltraFineWeb
    </div>
    <details class="paper-abstract">
      Data quality has become a key factor in enhancing model performance with the rapid development of large language models (LLMs). Model-driven data filtering has increasingly become a primary approach for acquiring high-quality data. However, it still faces two main challenges: (1) the lack of an efficient data verification strategy makes it difficult to provide timely feedback on data quality; and (2) the selection of seed data for training classifiers lacks clear criteria and relies heavily on human expertise, introducing a degree of subjectivity. To address the first challenge, we introduce an efficient verification strategy that enables rapid evaluation of the impact of data on LLM training with minimal computational cost. To tackle the second challenge, we build upon the assumption that high-quality seed data is beneficial for LLM training, and by integrating the proposed verification strategy, we optimize the selection of positive and negative samples and propose an efficient data filtering pipeline. This pipeline not only improves filtering efficiency, classifier quality, and robustness, but also significantly reduces experimental and inference costs. In addition, to efficiently filter high-quality data, we employ a lightweight classifier based on fastText, and successfully apply the filtering pipeline to two widely-used pre-training corpora, FineWeb and Chinese FineWeb datasets, resulting in the creation of the higher-quality Ultra-FineWeb dataset. Ultra-FineWeb contains approximately 1 trillion English tokens and 120 billion Chinese tokens. Empirical results demonstrate that the LLMs trained on Ultra-FineWeb exhibit significant performance improvements across multiple benchmark tasks, validating the effectiveness of our pipeline in enhancing both data quality and training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v1">TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 WIP
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce TransProQA, a novel, reference-free, LLM-based question-answering (QA) framework designed specifically for literary translation evaluation. TransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, TransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art (SOTA) metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, TransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05406v1">Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Framing in media critically shapes public perception by selectively emphasizing some details while downplaying others. With the rise of large language models in automated news and content creation, there is growing concern that these systems may introduce or even amplify framing biases compared to human authors. In this paper, we explore how framing manifests in both out-of-the-box and fine-tuned LLM-generated news content. Our analysis reveals that, particularly in politically and socially sensitive contexts, LLMs tend to exhibit more pronounced framing than their human counterparts. In addition, we observe significant variation in framing tendencies across different model architectures, with some models displaying notably higher biases. These findings point to the need for effective post-training mitigation strategies and tighter evaluation frameworks to ensure that automated news content upholds the standards of balanced reporting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02107v2">TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Code available at: https://github.com/apple/ml-tic-lm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07944v2">Enhancing Differential Testing With LLMs For Testing Deep Learning Libraries</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 This work has been accepted by ACM TOSEM. Manuscript under final preparation
    </div>
    <details class="paper-abstract">
      Differential testing offers a promising strategy to alleviate the test oracle problem by comparing the test results between alternative implementations. However, existing differential testing techniques for deep learning (DL) libraries are limited by the key challenges of finding alternative implementations (called counterparts) for a given API and subsequently generating diverse test inputs. To address the two challenges, this paper introduces DLLens, an LLM-enhanced differential testing technique for DL libraries. To address the first challenge, DLLens incorporates an LLM-based counterpart synthesis workflow, with the insight that the counterpart of a given DL library API's computation could be successfully synthesized through certain composition and adaptation of the APIs from another DL library. To address the second challenge, DLLens incorporates a static analysis technique that extracts the path constraints from the implementations of a given API and its counterpart to guide diverse test input generation. The extraction is facilitated by LLM's knowledge of the concerned DL library and its upstream libraries. We evaluate DLLens on two popular DL libraries, TensorFlow and PyTorch. Our evaluation shows that DLLens synthesizes counterparts for 1.84 times as many APIs as those found by state-of-the-art techniques on these libraries. Moreover, under the same time budget, DLLens covers 7.23% more branches and detects 1.88 times as many bugs as state-of-the-art techniques on 200 randomly sampled APIs. DLLens has successfully detected 71 bugs in recent TensorFlow and PyTorch libraries. Among them, 59 are confirmed by developers, including 46 confirmed as previously unknown bugs, and 10 of these previously unknown bugs have been fixed in the latest version of TensorFlow and PyTorch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05286v1">HEXGEN-TEXT2SQL: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL Workflow</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging the agentic paradigm of large language models (LLMs) utilization have significantly enhanced Text-to-SQL capabilities, enabling users without specialized database expertise to query data intuitively. However, deploying these agentic LLM-based Text-to-SQL systems in production poses substantial challenges due to their inherently multi-stage workflows, stringent latency constraints, and potentially heterogeneous GPU infrastructure in enterprise environments. Current LLM serving frameworks lack effective mechanisms for handling interdependent inference tasks, dynamic latency variability, and resource heterogeneity, leading to suboptimal performance and frequent service-level objective (SLO) violations. In this paper, we introduce HEXGEN-TEXT2SQL, a novel framework designed explicitly to schedule and execute agentic multi-stage LLM-based Text-to-SQL workflows on heterogeneous GPU clusters that handle multi-tenant end-to-end queries. HEXGEN-TEXT2SQL introduce a hierarchical scheduling approach combining global workload-balanced task dispatching and local adaptive urgency-guided prioritization, guided by a systematic analysis of agentic Text-to-SQL workflows. Additionally, we propose a lightweight simulation-based method for tuning critical scheduling hyperparameters, further enhancing robustness and adaptability. Our extensive evaluation on realistic Text-to-SQL benchmarks demonstrates that HEXGEN-TEXT2SQL significantly outperforms state-of-the-art LLM serving frameworks. Specifically, HEXGEN-TEXT2SQL reduces latency deadlines by up to 1.67$\times$ (average: 1.41$\times$) and improves system throughput by up to 1.75$\times$ (average: 1.65$\times$) compared to vLLM under diverse, realistic workload conditions. Our code is available at https://github.com/Relaxed-System-Lab/Hexgen-Flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05237v1">Latte: Transfering LLMs` Latent-level Knowledge for Few-shot Tabular Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Few-shot tabular learning, in which machine learning models are trained with a limited amount of labeled data, provides a cost-effective approach to addressing real-world challenges. The advent of Large Language Models (LLMs) has sparked interest in leveraging their pre-trained knowledge for few-shot tabular learning. Despite promising results, existing approaches either rely on test-time knowledge extraction, which introduces undesirable latency, or text-level knowledge, which leads to unreliable feature engineering. To overcome these limitations, we propose Latte, a training-time knowledge extraction framework that transfers the latent prior knowledge within LLMs to optimize a more generalized downstream model. Latte enables general knowledge-guided downstream tabular learning, facilitating the weighted fusion of information across different feature values while reducing the risk of overfitting to limited labeled data. Furthermore, Latte is compatible with existing unsupervised pre-training paradigms and effectively utilizes available unlabeled samples to overcome the performance limitations imposed by an extremely small labeled dataset. Extensive experiments on various few-shot tabular learning benchmarks demonstrate the superior performance of Latte, establishing it as a state-of-the-art approach in this domain
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05225v1">QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      The rapid advancement of Chinese large language models (LLMs) underscores the need for domain-specific evaluations to ensure reliable applications. However, existing benchmarks often lack coverage in vertical domains and offer limited insights into the Chinese working context. Leveraging qualification exams as a unified framework for human expertise evaluation, we introduce QualBench, the first multi-domain Chinese QA benchmark dedicated to localized assessment of Chinese LLMs. The dataset includes over 17,000 questions across six vertical domains, with data selections grounded in 24 Chinese qualifications to closely align with national policies and working standards. Through comprehensive evaluation, the Qwen2.5 model outperformed the more advanced GPT-4o, with Chinese LLMs consistently surpassing non-Chinese models, highlighting the importance of localized domain knowledge in meeting qualification requirements. The best performance of 75.26% reveals the current gaps in domain coverage within model capabilities. Furthermore, we present the failure of LLM collaboration with crowdsourcing mechanisms and suggest the opportunities for multi-domain RAG knowledge enhancement and vertical domain LLM training with Federated Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v3">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05196v1">Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      We present a systematic study of provider-side data poisoning in retrieval-augmented recommender systems (RAG-based). By modifying only a small fraction of tokens within item descriptions -- for instance, adding emotional keywords or borrowing phrases from semantically related items -- an attacker can significantly promote or demote targeted items. We formalize these attacks under token-edit and semantic-similarity constraints, and we examine their effectiveness in both promotion (long-tail items) and demotion (short-head items) scenarios. Our experiments on MovieLens, using two large language model (LLM) retrieval modules, show that even subtle attacks shift final rankings and item exposures while eluding naive detection. The results underscore the vulnerability of RAG-based pipelines to small-scale metadata rewrites and emphasize the need for robust textual consistency checks and provenance tracking to thwart stealthy provider-side poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02116v3">Advancements and limitations of LLMs in replicating human color-word associations</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 20 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Color-word associations play a fundamental role in human cognition and design applications. Large Language Models (LLMs) have become widely available and have demonstrated intelligent behaviors in various benchmarks with natural conversation skills. However, their ability to replicate human color-word associations remains understudied. We compared multiple generations of LLMs (from GPT-3 to GPT-4o) against human color-word associations using data collected from over 10,000 Japanese participants, involving 17 colors and 80 words (10 word from eight categories) in Japanese. Our findings reveal a clear progression in LLM performance across generations, with GPT-4o achieving the highest accuracy in predicting the best voted word for each color and category. However, the highest median performance was approximately 50% even for GPT-4o with visual inputs (chance level of 10%). Moreover, we found performance variations across word categories and colors: while LLMs tended to excel in categories such as Rhythm and Landscape, they struggled with categories such as Emotions. Interestingly, color discrimination ability estimated from our color-word association data showed high correlation with human color discrimination patterns, consistent with previous studies. Thus, despite reasonable alignment in basic color discrimination, humans and LLMs still diverge systematically in the words they assign to those colors. Our study highlights both the advancements in LLM capabilities and their persistent limitations, raising the possibility of systematic differences in semantic memory structures between humans and LLMs in representing color-word associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05040v3">SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Our code, data, and model will be released at https://github.com/InternLM/SWE-Fixer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency across a variety of complex tasks. One significant application of LLMs is in tackling software engineering challenges, particularly in resolving real-world tasks on GitHub by fixing code based on the issues reported by the users. However, many current approaches rely on proprietary LLMs, which limits reproducibility, accessibility, and transparency. The critical components of LLMs for addressing software engineering issues and how their capabilities can be effectively enhanced remain unclear. To address these challenges, we introduce SWE-Fixer, a novel open-source framework designed to effectively and efficiently resolve GitHub issues. SWE-Fixer comprises two essential modules: a code file retrieval module and a code editing module. The retrieval module employs BM25 along with a lightweight model to achieve coarse-to-fine file retrieval. Subsequently, the code editing module utilizes the other model to generate patches for the identified files. To mitigate the lack of publicly available datasets, we compile an extensive dataset that includes 110K GitHub issues along with their corresponding patches and train the two models of SWE-Fixer separately. We assess our approach on the SWE-Bench Lite and Verified benchmarks, achieving competitive performance among open-source models with scores of 22.0% and 30.2%. Furthermore, SWE-Fixer reaches state-of-the-art performance (24.7% on Lite and 32.8% on Verified) with PASS_TO_PASS (P2P) filtering. Additionally, our approach requires only two model calls per instance, making it significantly more efficient than existing methods. These results highlight the effectiveness of SWE-Fixer in real-world code-fixing scenarios. We will make our model, dataset, and code publicly available at https://github.com/InternLM/SWE-Fixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04101v1">LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are almost nonexistent studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11026v2">SceneLLM: Implicit Language Reasoning in LLM for Dynamic Scene Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 29 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Dynamic scenes contain intricate spatio-temporal information, crucial for mobile robots, UAVs, and autonomous driving systems to make informed decisions. Parsing these scenes into semantic triplets <Subject-Predicate-Object> for accurate Scene Graph Generation (SGG) is highly challenging due to the fluctuating spatio-temporal complexity. Inspired by the reasoning capabilities of Large Language Models (LLMs), we propose SceneLLM, a novel framework that leverages LLMs as powerful scene analyzers for dynamic SGG. Our framework introduces a Video-to-Language (V2L) mapping module that transforms video frames into linguistic signals (scene tokens), making the input more comprehensible for LLMs. To better encode spatial information, we devise a Spatial Information Aggregation (SIA) scheme, inspired by the structure of Chinese characters, which encodes spatial data into tokens. Using Optimal Transport (OT), we generate an implicit language signal from the frame-level token sequence that captures the video's spatio-temporal information. To further improve the LLM's ability to process this implicit linguistic input, we apply Low-Rank Adaptation (LoRA) to fine-tune the model. Finally, we use a transformer-based SGG predictor to decode the LLM's reasoning and predict semantic triplets. Our method achieves state-of-the-art results on the Action Genome (AG) benchmark, and extensive experiments show the effectiveness of SceneLLM in understanding and generating accurate dynamic scene graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04075v1">LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress?</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      This paper examines whether large language model (LLM) capabilities can continue to advance without additional compute by analyzing the development and role of algorithms used in state-of-the-art LLMs. Motivated by regulatory efforts that have largely focused on restricting access to high-performance hardware, we ask: Can LLMs progress in a compute-constrained environment, and how do algorithmic innovations perform under such conditions? To address these questions, we introduce a novel classification framework that distinguishes between compute-dependent innovations -- which yield disproportionate benefits at high compute levels (e.g., the Transformer architecture and mixture-of-experts models) and compute-independent innovations, which improve efficiency across all compute scales (e.g., rotary positional encoding, FlashAttention, or layer normalization). We quantify these contributions using a metric called compute-equivalent gain (CEG), which estimates the additional compute that would be required to achieve similar improvements without these algorithmic advancements. To validate this framework, we conduct small-scale training experiments with a scaled-down GPT-2 model. Our results confirm that compute-independent advancements yield meaningful performance gains even in resource-constrained settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast, compute-dependent advancements provided little benefit or even degraded performance at the small scale, reinforcing the importance of compute availability for certain algorithmic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04072v1">Advancing and Benchmarking Personalized Tool Invocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 14 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at https://github.com/hyfshadow/PTBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v2">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04847v1">Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04842v1">Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Prevalent reinforcement learning~(RL) methods for fine-tuning LLM reasoners, such as GRPO or Leave-one-out PPO, abandon the learned value function in favor of empirically estimated returns. This hinders test-time compute scaling that relies on using the value-function for verification. In this work, we propose RL$^V$ that augments any ``value-free'' RL method by jointly training the LLM as both a reasoner and a generative verifier using RL-generated data, adding verification capabilities without significant overhead. Empirically, RL$^V$ boosts MATH accuracy by over 20\% with parallel sampling and enables $8-32\times$ efficient test-time compute scaling compared to the base RL method. RL$^V$ also exhibits strong generalization capabilities for both easy-to-hard and out-of-domain tasks. Furthermore, RL$^V$ achieves $1.2-1.6\times$ higher performance when jointly scaling parallel and sequential test-time compute with a long reasoning R1 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04806v1">Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 7 Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04736v1">The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04732v1">QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04723v1">SOAEsV2-7B/72B: Full-Pipeline Optimization for State-Owned Enterprise LLMs via Continual Pre-Training, Domain-Progressive SFT and Distillation-Enhanced Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      This study addresses key challenges in developing domain-specific large language models (LLMs) for Chinese state-owned assets and enterprises (SOAEs), where current approaches face three limitations: 1) constrained model capacity that limits knowledge integration and cross-task adaptability; 2) excessive reliance on domain-specific supervised fine-tuning (SFT) data, which neglects the broader applicability of general language patterns; and 3) inefficient inference acceleration for large models processing long contexts. In this work, we propose SOAEsV2-7B/72B, a specialized LLM series developed via a three-phase framework: 1) continual pre-training integrates domain knowledge while retaining base capabilities; 2) domain-progressive SFT employs curriculum-based learning strategy, transitioning from weakly relevant conversational data to expert-annotated SOAEs datasets to optimize domain-specific tasks; 3) distillation-enhanced speculative decoding accelerates inference via logit distillation between 72B target and 7B draft models, achieving 1.39-1.52$\times$ speedup without quality loss. Experimental results demonstrate that our domain-specific pre-training phase maintains 99.8% of original general language capabilities while significantly improving domain performance, resulting in a 1.08$\times$ improvement in Rouge-1 score and a 1.17$\times$ enhancement in BLEU-4 score. Ablation studies further show that domain-progressive SFT outperforms single-stage training, achieving 1.02$\times$ improvement in Rouge-1 and 1.06$\times$ in BLEU-4. Our work introduces a comprehensive, full-pipeline approach for optimizing SOAEs LLMs, bridging the gap between general language capabilities and domain-specific expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04710v1">Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      With the increasing popularity of IoT applications, end users demand more personalized and intuitive functionality. A major obstacle for this, however, is that custom IoT functionality today still requires at least some coding skills. To address this, no-code development platforms have been proposed as a solution for empowering non-technical users to create applications. However, such platforms still require a certain level of technical expertise for structuring process steps or defining event-action relations. The advent of LLMs can further enhance no-code platforms by enabling natural language-based interaction, automating of complex tasks, and dynamic code generation. By allowing users to describe their requirements in natural language, LLMs can significantly streamline no-code development. As LLMs vary in performance, architecture, training data used, and the use cases they target, it is still unclear which models are best suited and what are the influence factors determining this fit. In particular, no-code development of IoT applications by non-technical users will have completely different demands on LLMs than, e.g., code generation for more open-ended applications or for supporting professional developers. In this paper, we explore the factors influencing the suitability of LLMs to no-code development of IoT applications. We also examine the role of input prompt language on accuracy and quality of generated applications as well as the influence of LLM training data. By conducting comprehensive experiments with a range of LLMs, we provide valuable insights for optimizing LLM-powered no-code platforms, guiding the selection of the suitable LLMs and their effective application. Our findings contribute to improving the accessibility, efficiency, and user experience of no-code IoT development, ultimately enabling broader adoption of IoT technologies among non-expert users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04673v1">REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages (8 main), to be published in IJCAI 2025
    </div>
    <details class="paper-abstract">
      Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o. We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04670v1">LLM Code Customization with Visual Results: A Benchmark on TikZ</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04660v1">AI-Generated Fall Data: Assessing LLMs and Diffusion Model for Wearable Fall Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Training fall detection systems is challenging due to the scarcity of real-world fall data, particularly from elderly individuals. To address this, we explore the potential of Large Language Models (LLMs) for generating synthetic fall data. This study evaluates text-to-motion (T2M, SATO, ParCo) and text-to-text models (GPT4o, GPT4, Gemini) in simulating realistic fall scenarios. We generate synthetic datasets and integrate them with four real-world baseline datasets to assess their impact on fall detection performance using a Long Short-Term Memory (LSTM) model. Additionally, we compare LLM-generated synthetic data with a diffusion-based method to evaluate their alignment with real accelerometer distributions. Results indicate that dataset characteristics significantly influence the effectiveness of synthetic data, with LLM-generated data performing best in low-frequency settings (e.g., 20Hz) while showing instability in high-frequency datasets (e.g., 200Hz). While text-to-motion models produce more realistic biomechanical data than text-to-text models, their impact on fall detection varies. Diffusion-based synthetic data demonstrates the closest alignment to real data but does not consistently enhance model performance. An ablation study further confirms that the effectiveness of synthetic data depends on sensor placement and fall representation. These findings provide insights into optimizing synthetic data generation for fall detection models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07852v1">Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04588v1">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a reinforcement learning framework that incentivizes the search capabilities of LLMs without interacting with real search engines. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both relevant and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
    </details>
</div>
