# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
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
- [Part 20](papers_20.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v3">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19264v1">LAPRAD: LLM-Assisted PRotocol Attack Discovery</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 IFIP Networking 2025 Proceedings (Accepted on 05.05.2025)
    </div>
    <details class="paper-abstract">
      With the goal of improving the security of Internet protocols, we seek faster, semi-automatic methods to discover new vulnerabilities in protocols such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers with some DNS knowledge to efficiently uncover vulnerabilities that would otherwise be hard to detect. LAPRAD follows a three-stage process. In the first, we consult an LLM (GPT-o1) that has been trained on a broad corpus of DNS-related sources and previous DDoS attacks to identify potential exploits. In the second stage, a different LLM automatically constructs the corresponding attack configurations using the ReACT approach implemented via LangChain (DNS zone file generation). Finally, in the third stage, we validate the attack's functionality and effectiveness. Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and rediscovered two recently reported ones that were not included in the LLM's training data. The first new attack employs a bait-and-switch technique to trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving capacity to as little as 6%. The second exploits large DNSSEC encryption algorithms (RSA-4096) with multiple keys, thereby bypassing a recently implemented default RRSet limit. The third leverages ANY-type responses to produce a similar effect. These variations of a cache-flushing DDoS attack, called SigCacheFlush, circumvent existing patches, severely degrade resolver query capacity, and impact the latest versions of major DNS resolver implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19252v1">LLMartini: Seamless and Interactive Leveraging of Multiple LLMs through Comparison and Composition</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The growing diversity of large language models (LLMs) means users often need to compare and combine outputs from different models to obtain higher-quality or more comprehensive responses. However, switching between separate interfaces and manually integrating outputs is inherently inefficient, leading to a high cognitive burden and fragmented workflows. To address this, we present LLMartini, a novel interactive system that supports seamless comparison, selection, and intuitive cross-model composition tools. The system decomposes responses into semantically aligned segments based on task-specific criteria, automatically merges consensus content, and highlights model differences through color coding while preserving unique contributions. In a user study (N=18), LLMartini significantly outperformed conventional manual methods across all measured metrics, including task completion time, cognitive load, and user satisfaction. Our work highlights the importance of human-centered design in enhancing the efficiency and creativity of multi-LLM interactions and offers practical implications for leveraging the complementary strengths of various language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18179v2">Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 13 pages, 8 figures. Accepted for presentation at the 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: https://github.com/AdCo-Research/adaptive-coopetition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06252v2">ZipLLM: Efficient LLM Storage via Model-Aware Synergistic Data Deduplication and Compression</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Modern model hubs, such as Hugging Face, store tens of petabytes of LLMs, with fine-tuned variants vastly outnumbering base models and dominating storage consumption. Existing storage reduction techniques -- such as deduplication and compression -- are either LLM-oblivious or not compatible with each other, limiting data reduction effectiveness. Our large-scale characterization study across all publicly available Hugging Face LLM repositories reveals several key insights: (1) fine-tuned models within the same family exhibit highly structured, sparse parameter differences suitable for delta compression; (2) bitwise similarity enables LLM family clustering; and (3) tensor-level deduplication is better aligned with model storage workloads, achieving high data reduction with low metadata overhead. Building on these insights, we design BitX, an effective, fast, lossless delta compression algorithm that compresses XORed difference between fine-tuned and base LLMs. We build ZipLLM, a model storage reduction pipeline that unifies tensor-level deduplication and lossless BitX compression. By synergizing deduplication and compression around LLM family clustering, ZipLLM reduces model storage consumption by 54%, over 20% higher than state-of-the-art deduplication and compression approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19225v1">RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located ones fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout. In this paper, we present RLBoost, a systematic solution for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18795v2">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19208v1">DiSRouter: Distributed Self-Routing for LLM Selections</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v4">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 EMNLP 2025 (Oral, ARR best paper nomination)
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15831v2">Who's Asking? Investigating Bias Through the Lens of Disability Framed Queries in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) routinely infer users demographic traits from phrasing alone, which can result in biased responses, even when no explicit demographic information is provided. The role of disability cues in shaping these inferences remains largely uncharted. Thus, we present the first systematic audit of disability-conditioned demographic bias across eight state-of-the-art instruction-tuned LLMs ranging from 3B to 72B parameters. Using a balanced template corpus that pairs nine disability categories with six real-world business domains, we prompt each model to predict five demographic attributes - gender, socioeconomic status, education, cultural background, and locality - under both neutral and disability-aware conditions. Across a varied set of prompts, models deliver a definitive demographic guess in up to 97\% of cases, exposing a strong tendency to make arbitrary inferences with no clear justification. Disability context heavily shifts predicted attribute distributions, and domain context can further amplify these deviations. We observe that larger models are simultaneously more sensitive to disability cues and more prone to biased reasoning, indicating that scale alone does not mitigate stereotype amplification. Our findings reveal persistent intersections between ableism and other demographic stereotypes, pinpointing critical blind spots in current alignment strategies. We release our evaluation framework and results to encourage disability-inclusive benchmarking and recommend integrating abstention calibration and counterfactual fine-tuning to curb unwarranted demographic inference. Code and data will be released on acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19178v1">Imbalanced Gradients in RL Post-Training of Multi-Task LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Multi-task post-training of large language models (LLMs) is typically performed by mixing datasets from different tasks and optimizing them jointly. This approach implicitly assumes that all tasks contribute gradients of similar magnitudes; when this assumption fails, optimization becomes biased toward large-gradient tasks. In this paper, however, we show that this assumption fails in RL post-training: certain tasks produce significantly larger gradients, thus biasing updates toward those tasks. Such gradient imbalance would be justified only if larger gradients implied larger learning gains on the tasks (i.e., larger performance improvements) -- but we find this is not true. Large-gradient tasks can achieve similar or even much lower learning gains than small-gradient ones. Further analyses reveal that these gradient imbalances cannot be explained by typical training statistics such as training rewards or advantages, suggesting that they arise from the inherent differences between tasks. This cautions against naive dataset mixing and calls for future work on principled gradient-level corrections for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12499v2">PTFA: An LLM-based Agent that Facilitates Online Consensus Building through Parallel Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-22
    </div>
    <details class="paper-abstract">
      Consensus building is inherently challenging due to the diverse opinions held by stakeholders. Effective facilitation is crucial to support the consensus building process and enable efficient group decision making. However, the effectiveness of facilitation is often constrained by human factors such as limited experience and scalability. In this research, we propose a Parallel Thinking-based Facilitation Agent (PTFA) that facilitates online, text-based consensus building processes.The PTFA automatically collects real-time textual input and leverages large language models (LLMs)to perform all six distinct roles of the well-established Six Thinking Hats technique in parallel thinking.To illustrate the potential of the agent, a pilot study was conducted, demonstrating its capabilities in idea generation, emotional probing, and deeper analysis of idea quality. Additionally, future open research challenges such as optimizing scheduling and managing behaviors in divergent phase are identified. Furthermore, a comprehensive dataset that contains not only the conversational content among the participants but also between the participants and the agent is constructed for future study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12631v2">Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 This work has been accepted to DAI 2025
    </div>
    <details class="paper-abstract">
      Balancing performance and efficiency is a central challenge in large language model (LLM) advancement. GPT-5 addresses this with test-time routing, dynamically assigning queries to either an efficient or a high-capacity model during inference. In this work, we present Avengers-Pro, a test-time routing framework that ensembles LLMs of varying capacities and efficiencies, providing a unified solution for all performance-efficiency tradeoffs. The Avengers-Pro embeds and clusters incoming queries, then routes each to the most suitable model based on a performance-efficiency score. Across 6 challenging benchmarks and 8 leading models -- including GPT-5-medium, Gemini-2.5-pro, and Claude-opus-4.1 -- Avengers-Pro achieves state-of-the-art results: by varying a performance-efficiency trade-off parameter, it can surpass the strongest single model (GPT-5-medium) by +7% in average accuracy. Moreover, it can match the average accuracy of the strongest single model at 27% lower cost, and reach ~90% of that performance at 63% lower cost. Last but not least, it achieves a Pareto frontier, consistently yielding the highest accuracy for any given cost, and the lowest cost for any given accuracy, among all single models. Code is available at https://github.com/ZhangYiqun018/AvengersPro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19172v1">When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18279v2">Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Accepted to EMNLP 2025 Findings ("Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs")
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18527v2">LLMs as Sparse Retrievers:A Framework for First-Stage Product Search</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Product search is a crucial component of modern e-commerce platforms, with billions of user queries every day. In product search systems, first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval is particularly attractive in this context due to its interpretability and storage efficiency. However, sparse retrieval methods suffer from severe vocabulary mismatch issues, leading to suboptimal performance in product search scenarios. With their potential for semantic analysis, large language models (LLMs) offer a promising avenue for mitigating vocabulary mismatch issues and thereby improving retrieval quality. Directly applying LLMs to sparse retrieval in product search exposes two key challenges:(1)Queries and product titles are typically short and highly susceptible to LLM-induced hallucinations, such as generating irrelevant expansion terms or underweighting critical literal terms like brand names and model numbers;(2)The large vocabulary space of LLMs leads to difficulty in initializing training effectively, making it challenging to learn meaningful sparse representations in such ultra-high-dimensional spaces.To address these challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that alleviates hallucination in lexical expansion by reinforcing underweighted literal terms through a residual compensation mechanism; and (2)A lexical focusing window that facilitates effective training initialization via a coarse-to-fine sparsification strategy.Extensive offline and online experiments show that PROSPER significantly outperforms sparse baselines and achieves recall performance comparable to advanced dense retrievers, while also achieving revenue increments online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v4">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 NeurIPS Camera-Ready Version. Code available at: https://github.com/Graph-COM/Knowledge_Unlearning
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18585v1">CLASP: Cost-Optimized LLM-based Agentic System for Phishing Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted in the 5th International Conference on Electrical, Computer, and Energy Technologies (ICECET2025)
    </div>
    <details class="paper-abstract">
      Phishing websites remain a significant cybersecurity threat, necessitating accurate and cost-effective detection mechanisms. In this paper, we present CLASP, a novel system that effectively identifies phishing websites by leveraging multiple intelligent agents, built using large language models (LLMs), to analyze different aspects of a web resource. The system processes URLs or QR codes, employing specialized LLM-based agents that evaluate the URL structure, webpage screenshot, and HTML content to predict potential phishing threats. To optimize performance while minimizing operational costs, we experimented with multiple combination strategies for agent-based analysis, ultimately designing a strategic combination that ensures the per-website evaluation expense remains minimal without compromising detection accuracy. We tested various LLMs, including Gemini 1.5 Flash and GPT-4o mini, to build these agents and found that Gemini 1.5 Flash achieved the best performance with an F1 score of 83.01% on a newly curated dataset. Also, the system maintained an average processing time of 2.78 seconds per website and an API cost of around $3.18 per 1,000 websites. Moreover, CLASP surpasses leading previous solutions, achieving over 40% higher recall and a 20% improvement in F1 score for phishing detection on the collected dataset. To support further research, we have made our dataset publicly available, supporting the development of more advanced phishing detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18560v1">WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The paradigm of LLM-as-a-judge is emerging as a scalable and efficient alternative to human evaluation, demonstrating strong performance on well-defined tasks. However, its reliability in open-ended tasks with dynamic environments and complex interactions remains unexplored. To bridge the gap, we introduce WebDevJudge, a systematic benchmark for assessing LLM-as-a-judge performance in web development, with support for both non-interactive evaluation based on static observations and continuous interactive evaluation with a dynamic web environment. WebDevJudge comprises human preference labels over paired web implementations, annotated with structured and query-grounded rubrics to ensure high-quality ground truth. Using this benchmark, we comprehensively evaluate various evaluators, including LLMs, MLLMs, and agentic workflows. We systematically investigate the impact of different paradigms and guidance mechanisms. Our experiments reveal a significant gap between LLM judges and human experts. In-depth analysis indicates this gap stems from fundamental model limitations, including failures in recognizing functional equivalence, verifying task feasibility, and mitigating bias. Overall, WebDevJudge presents a significant challenge to LLM-as-a-judge, offering insights to guide future research toward developing more reliable and capable automated evaluators for complicated scenarios. Code and data are available at https://github.com/lcy2723/WebDevJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18550v1">JAUNT: Joint Alignment of User Intent and Network State for QoE-centric LLM Tool Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on emerging protocols such as the Model Context Protocol (MCP) to invoke external tools and services. However, current tool routing mechanisms remain fragile because they only consider functional matching between users' queries and tools. In practice, user intent expressed through queries can be vague or underspecified, and the actual Quality of Experience (QoE) also depends on external factors such as link latency and server availability that are not captured by semantics alone. To address this challenge, we propose JAUNT, a framework for Joint Alignment of User intent and Network state in QoE-centric Tool routing. JAUNT introduces a dual-view alignment strategy that interprets user intent while employing LLM agents to construct network profiles, mapping numerical performance indicators into the semantic space to guide routing. We further design a benchmark that integrates diverse user request patterns with heterogeneous network states, enabling systematic evaluation of QoE outcomes. Experimental results show that JAUNT significantly improves QoE compared with several baselines, demonstrating the importance of aligning both intent and network state for scalable LLM service orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18544v1">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18527v1">LLMs as Sparse Retrievers:A Framework for First-Stage Product Search</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Product search is a crucial component of modern e-commerce platforms, with billions of user queries every day. In product search systems, first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval is particularly attractive in this context due to its interpretability and storage efficiency. However, sparse retrieval methods suffer from severe vocabulary mismatch issues, leading to suboptimal performance in product search scenarios.With their potential for semantic analysis, large language models (LLMs) offer a promising avenue for mitigating vocabulary mismatch issues and thereby improving retrieval quality. Directly applying LLMs to sparse retrieval in product search exposes two key challenges:(1)Queries and product titles are typically short and highly susceptible to LLM-induced hallucinations, such as generating irrelevant expansion terms or underweighting critical literal terms like brand names and model numbers;(2)The large vocabulary space of LLMs leads to difficulty in initializing training effectively, making it challenging to learn meaningful sparse representations in such ultra-high-dimensional spaces.To address these challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that alleviates hallucination in lexical expansion by reinforcing underweighted literal terms through a residual compensation mechanism; and (2)A lexical focusing window that facilitates effective training initialization via a coarse-to-fine sparsification strategy.Extensive offline and online experiments show that PROSPER significantly outperforms sparse baselines and achieves recall performance comparable to advanced dense retrievers, while also achieving revenue increments online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18525v1">From Quarter to All: Accelerating Speculative LLM Decoding via Floating-Point Exponent Remapping and Parameter Sharing</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language models achieve impressive performance across diverse tasks but exhibit high inference latency due to their large parameter sizes. While quantization reduces model size, it often leads to performance degradation compared to the full model. Speculative decoding remains lossless but typically incurs extra overheads. We propose SPEQ, an algorithm-hardware co-designed speculative decoding method that uses part of the full-model weight bits to form a quantized draft model, thereby eliminating additional training or storage overhead. A reconfigurable processing element array enables efficient execution of both the draft and verification passes. Experimental results across 15 LLMs and tasks demonstrate that SPEQ achieves speedups of 2.07x, 1.53x, and 1.45x compared over FP16, Olive, and Tender, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14045v4">From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 EMNLP 2025 Main Conference (Oral)
    </div>
    <details class="paper-abstract">
      Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18508v1">Prompting the Priorities: A First Look at Evaluating LLMs for Vulnerability Triage and Prioritization</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Security analysts face increasing pressure to triage large and complex vulnerability backlogs. Large Language Models (LLMs) offer a potential aid by automating parts of the interpretation process. We evaluate four models (ChatGPT, Claude, Gemini, and DeepSeek) across twelve prompting techniques to interpret semi-structured and unstructured vulnerability information. As a concrete use case, we test each model's ability to predict decision points in the Stakeholder-Specific Vulnerability Categorization (SSVC) framework: Exploitation, Automatable, Technical Impact, and Mission and Wellbeing. Using 384 real-world vulnerabilities from the VulZoo dataset, we issued more than 165,000 queries to assess performance under prompting styles including one-shot, few-shot, and chain-of-thought. We report F1 scores for each SSVC decision point and Cohen's kappa (weighted and unweighted) for the final SSVC decision outcomes. Gemini consistently ranked highest, leading on three of four decision points and yielding the most correct recommendations. Prompting with exemplars generally improved accuracy, although all models struggled on some decision points. Only DeepSeek achieved fair agreement under weighted metrics, and all models tended to over-predict risk. Overall, current LLMs do not replace expert judgment. However, specific LLM and prompt combinations show moderate effectiveness for targeted SSVC decisions. When applied with care, LLMs can support vulnerability prioritization workflows and help security teams respond more efficiently to emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18491v1">Crucible: Quantifying the Potential of Control Algorithms through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Control algorithms in production environments typically require domain experts to tune their parameters and logic for specific scenarios. However, existing research predominantly focuses on algorithmic performance under ideal or default configurations, overlooking the critical aspect of Tuning Potential. To bridge this gap, we introduce Crucible, an agent that employs an LLM-driven, multi-level expert simulation to turn algorithms and defines a formalized metric to quantitatively evaluate their Tuning Potential. We demonstrate Crucible's effectiveness across a wide spectrum of case studies, from classic control tasks to complex computer systems, and validate its findings in a real-world deployment. Our experimental results reveal that Crucible systematically quantifies the tunable space across different algorithms. Furthermore, Crucible provides a new dimension for algorithm analysis and design, which ultimately leads to performance improvements. Our code is available at https://github.com/thu-media/Crucible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18477v1">LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18476v1">Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We present a probabilistic intent modeling framework for large language model (LLM) agents in multi-turn social dialogue. The framework maintains a belief distribution over a partner's latent intentions, initialized from contextual priors and dynamically updated through likelihood estimation after each utterance. The evolving distribution provides additional contextual grounding for the policy, enabling adaptive dialogue strategies under uncertainty. Preliminary experiments in the SOTOPIA environment show consistent improvements: the proposed framework increases the Overall score by 9.0% on SOTOPIA-All and 4.1% on SOTOPIA-Hard compared with the Qwen2.5-7B baseline, and slightly surpasses an oracle agent that directly observes partner intentions. These early results suggest that probabilistic intent modeling can contribute to the development of socially intelligent LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18470v1">CircuitSeer: Mining High-Quality Data by Probing Mathematical Reasoning Circuits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities, but scaling their performance often relies on massive reasoning datasets that are computationally expensive to train on. Existing data selection methods aim to curate smaller, high-quality subsets but often rely on costly external models or opaque heuristics. In this work, we shift the focus from external heuristics to the model's internal mechanisms. We find that complex reasoning tasks consistently activate a sparse, specialized subset of attention heads, forming core reasoning circuits. Building on this insight, we propose CircuitSeer, a novel data selection method that quantifies the reasoning complexity of data by measuring its influence on these crucial circuits. Extensive experiments on 4 models and 9 datasets demonstrate CircuitSeer's superiority. Notably, fine-tuning Qwen2.5-Math-7B on just 10% of data selected by our method achieves a 1.4-point gain in average Pass@1 over training on the full dataset, highlighting its efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18438v1">DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted to ASE'25
    </div>
    <details class="paper-abstract">
      Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18428v1">AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Optimization modeling enables critical decisions across industries but remains difficult to automate: informal language must be mapped to precise mathematical formulations and executable solver code. Prior LLM approaches either rely on brittle prompting or costly retraining with limited generalization. We present AlphaOPT, a self-improving experience library that enables an LLM to learn from limited demonstrations (even answers alone, without gold-standard programs) and solver feedback - without annotated reasoning traces or parameter updates. AlphaOPT operates in a continual two-phase cycle: (i) a Library Learning phase that reflects on failed attempts, extracting solver-verified, structured insights as {taxonomy, condition, explanation, example}; and (ii) a Library Evolution phase that diagnoses retrieval misalignments and refines the applicability conditions of stored insights, improving transfer across tasks. This design (1) learns efficiently from limited demonstrations without curated rationales, (2) expands continually without costly retraining by updating the library rather than model weights, and (3) makes knowledge explicit and interpretable for human inspection and intervention. Experiments show that AlphaOPT steadily improves with more data (65% to 72% from 100 to 300 training items) and surpasses the strongest baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only on answers. Code and data are available at: https://github.com/Minw913/AlphaOPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18395v1">Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 10 pages, 4 figures, 1 table, 1 algorithm. Submitted to conference
    </div>
    <details class="paper-abstract">
      This paper proposes Memory-Augmented State Machine Prompting (MASMP), a novel framework for LLM agents in real-time strategy games. Addressing key challenges like hallucinations and fragmented decision-making in existing approaches, MASMP integrates state machine prompting with memory mechanisms to unify structured actions with long-term tactical coherence. The framework features: (1) a natural language-driven state machine architecture that guides LLMs to emulate finite state machines and behavior trees through prompts, and (2) a lightweight memory module preserving strategic variables (e.g., tactics, priority units) across decision cycles. Experiments in StarCraft II demonstrate MASMP's 60% win rate against the hardest built-in AI (Lv7), vastly outperforming baselines (0%). Case studies reveal the method retains LLMs' semantic comprehension while resolving the "Knowing-Doing Gap" through strict state-action mapping, achieving both interpretability and FSM-like reliability. This work establishes a new paradigm for combining neural and symbolic AI in complex decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18368v1">KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We present $\textbf{Korean SimpleQA (KoSimpleQA)}$, a benchmark for evaluating factuality in large language models (LLMs) with a focus on Korean cultural knowledge. KoSimpleQA is designed to be challenging yet easy to grade, consisting of 1,000 short, fact-seeking questions with unambiguous answers. We conduct a comprehensive evaluation across a diverse set of open-source LLMs of varying sizes that support Korean, and find that even the strongest model generates correct answer only 33.7% of the time, underscoring the challenging nature of KoSimpleQA. Notably, performance rankings on KoSimpleQA differ substantially from those on the English SimpleQA, highlighting the unique value of our dataset. Furthermore, our analysis of reasoning LLMs shows that engaging reasoning capabilities in the factual QA task can both help models better elicit their latent knowledge and improve their ability to abstain when uncertain. KoSimpleQA can be found at https://anonymous.4open.science/r/KoSimpleQA-62EB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18364v1">Evaluating LLM-Based Mobile App Recommendations: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to recommend mobile applications through natural language prompts, offering a flexible alternative to keyword-based app store search. Yet, the reasoning behind these recommendations remains opaque, raising questions about their consistency, explainability, and alignment with traditional App Store Optimization (ASO) metrics. In this paper, we present an empirical analysis of how widely-used general purpose LLMs generate, justify, and rank mobile app recommendations. Our contributions are: (i) a taxonomy of 16 generalizable ranking criteria elicited from LLM outputs; (ii) a systematic evaluation framework to analyse recommendation consistency and responsiveness to explicit ranking instructions; and (iii) a replication package to support reproducibility and future research on AI-based recommendation systems. Our findings reveal that LLMs rely on a broad yet fragmented set of ranking criteria, only partially aligned with standard ASO metrics. While top-ranked apps tend to be consistent across runs, variability increases with ranking depth and search specificity. LLMs exhibit varying sensitivity to explicit ranking instructions - ranging from substantial adaptations to near-identical outputs - highlighting their complex reasoning dynamics in conversational app discovery. Our results aim to support end-users, app developers, and recommender-systems researchers in navigating the emerging landscape of conversational app discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04150v3">Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) suffer from temporal misalignment issues especially across long span of time. The issue arises from knowing that LLMs are trained on large amounts of data where temporal information is rather sparse over long times, such as thousands of years, resulting in insufficient learning or catastrophic forgetting by the LLMs. This paper proposes a methodology named "Ticktack" for addressing the LLM's long-time span misalignment in a yearly setting. Specifically, we first propose to utilize the sexagenary year expression instead of the Gregorian year expression employed by LLMs, achieving a more uniform distribution in yearly granularity. Then, we employ polar coordinates to model the sexagenary cycle of 60 terms and the year order within each term, with additional temporal encoding to ensure LLMs understand them. Finally, we present a temporal representational alignment approach for post-training LLMs that effectively distinguishes time points with relevant knowledge, hence improving performance on time-related tasks, particularly over a long period. We also create a long time span benchmark for evaluation. Experimental results prove the effectiveness of our proposal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12024v3">FlexQuant: A Flexible and Efficient Dynamic Precision Switching Framework for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 10 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has exacerbated the memory bottleneck due to the widening gap between model parameter scaling and hardware capabilities. While post-training quantization techniques effectively reduce memory overhead, existing methods predominantly rely on static quantization strategies, which struggle to adapt to dynamic workloads. To address this, we propose FlexQuant, a dynamic precision-switching framework that optimizes the trade-off between inference speed and accuracy. Leveraging model perplexity entropy and Kullback-Leibler divergence, FlexQuant enables fine-grained, layer-wise mixed-precision quantization and dynamically adjusts bit-widths during each token generation. FlexQuant provides a comprehensive analysis of quantization strategies, introduces a precision requirement model for optimal switching, and implements efficient fine-grained precision management. Evaluations demonstrate that FlexQuant achieves a 1.3x end-to-end speedup across diverse language tasks with negligible accuracy loss introduced. This framework offers a flexible and adaptive solution for efficient LLM deployment. Code is released at https://github.com/ZongwuWang/FlexQuant.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20002v5">The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 This work was first submitted for review on Sept. 5, 2024, and the initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version has accepted for publication by IEEE Transactions on Information Forensics and Security (TIFS)
    </div>
    <details class="paper-abstract">
      The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18333v1">Position: LLM Watermarking Should Align Stakeholders' Incentives for Practical Adoption</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Despite progress in watermarking algorithms for large language models (LLMs), real-world deployment remains limited. We argue that this gap stems from misaligned incentives among LLM providers, platforms, and end users, which manifest as four key barriers: competitive risk, detection-tool governance, robustness concerns and attribution issues. We revisit three classes of watermarking through this lens. \emph{Model watermarking} naturally aligns with LLM provider interests, yet faces new challenges in open-source ecosystems. \emph{LLM text watermarking} offers modest provider benefit when framed solely as an anti-misuse tool, but can gain traction in narrowly scoped settings such as dataset de-contamination or user-controlled provenance. \emph{In-context watermarking} (ICW) is tailored for trusted parties, such as conference organizers or educators, who embed hidden watermarking instructions into documents. If a dishonest reviewer or student submits this text to an LLM, the output carries a detectable watermark indicating misuse. This setup aligns incentives: users experience no quality loss, trusted parties gain a detection tool, and LLM providers remain neutral by simply following watermark instructions. We advocate for a broader exploration of incentive-aligned methods, with ICW as an example, in domains where trusted parties need reliable tools to detect misuse. More broadly, we distill design principles for incentive-aligned, domain-specific watermarking and outline future research directions. Our position is that the practical adoption of LLM watermarking requires aligning stakeholder incentives in targeted application domains and fostering active community engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18327v1">InspectCoder: Dynamic Analysis-Enabled Self Repair through interactive LLM-Debugger Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate buggy code with complex logic errors that are challenging to diagnose. While existing LLM-based self-repair approaches conduct intensive static semantic analysis or reply on superficial execution logs, they miss the in-depth runtime behaviors that often expose bug root causes-lacking the interactive dynamic analysis capabilities that make human debugging effective. We present InspectCoder, the first agentic program repair system that empowers LLMs to actively conduct dynamic analysis via interactive debugger control. Our dual-agent framework enables strategic breakpoint placement, targeted state inspection, and incremental runtime experimentation within stateful debugger sessions. Unlike existing methods that follow fixed log collection procedures, InspectCoder adaptively inspects and perturbs relevant intermediate states at runtime, and leverages immediate process rewards from debugger feedback to guide multi-step reasoning, transforming LLM debugging paradigm from blind trial-and-error into systematic root cause diagnosis. We conduct comprehensive experiments on two challenging self-repair benchmarks: BigCodeBench-R and LiveCodeBench-R. InspectCoder achieves 5.10%-60.37% relative improvements in repair accuracy over the strongest baseline, while delivering 1.67x-2.24x superior bug-fix efficiency respectively. We also contribute InspectWare, an open-source middleware that abstracts debugger complexities and maintains stateful debugging sessions across mainstream Python testing frameworks. Our work provides actionable insight into the interactive LLM-debugger systems, demonstrating the significant potential of LLM-driven dynamic analysis for automated software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21015v2">Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders (CE), which require full corpus access. We propose a corpus-free alternative: an end-to-end pipeline where a Large Language Model (LLM) first generates a query from a passage and then produces a hard negative example using only the generated query text. Our dataset comprises 7,250 arXiv abstracts spanning diverse domains including mathematics, physics, computer science, and related fields, serving as positive passages for query generation. We evaluate two fine-tuning configurations of DistilBERT for dense retrieval; one using LLM-generated hard negatives conditioned solely on the query, and another using negatives generated with both the query and its positive document as context. Compared to traditional corpus-based mining methods {LLM Query $\rightarrow$ BM25 HN and LLM Query $\rightarrow$ CE HN on multiple BEIR benchmark datasets, our all-LLM pipeline outperforms strong lexical mining baselines and achieves performance comparable to cross-encoder-based methods, demonstrating the potential of corpus-free hard negative generation for retrieval model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v5">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18314v1">Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17173v2">Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted to the NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      We study a web-deployed, tool-augmented LLM health coach with real users. In a pilot with seven users (280 rated turns), offline policy evaluation (OPE) over factorized decision heads (Tool/Style) shows that a uniform heavy-tool policy raises average value on logs but harms specific subgroups, most notably low-health-literacy/high-self-efficacy users. A lightweight simulator with hidden archetypes further shows that adding a small early information-gain bonus reliably shortens trait identification and improves goal success and pass@3. Together, these early findings indicate an evaluation-first path to personalization: freeze the generator, learn subgroup-aware decision heads on typed rewards (objective tool outcomes and satisfaction), and always report per-archetype metrics to surface subgroup harms that averages obscure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17727v2">Enabling Fine-Grained Operating Points for Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Under review at ICLR 2026. 36 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Black-box Large Language Models (LLMs) provide practical and accessible alternatives to other machine learning methods, as they require minimal labeled data and machine learning expertise to develop solutions for various decision making problems. However, for applications that need operating with constraints on specific metrics (e.g., precision $\geq$ 95%), decision making with black-box LLMs remains unfavorable, due to their low numerical output cardinalities. This results in limited control over their operating points, preventing fine-grained adjustment of their decision making behavior. In this paper, we study using black-box LLMs as classifiers, focusing on efficiently improving their operational granularity without performance loss. Specifically, we first investigate the reasons behind their low-cardinality numerical outputs and show that they are biased towards generating rounded but informative verbalized probabilities. Then, we experiment with standard prompt engineering, uncertainty estimation and confidence elicitation techniques, and observe that they do not effectively improve operational granularity without sacrificing performance or increasing inference cost. Finally, we propose efficient approaches to significantly increase the number and diversity of available operating points. Our proposed approaches provide finer-grained operating points and achieve comparable to or better performance than the benchmark methods across 11 datasets and 3 LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10255v2">When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 2nd version update to Jun.2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, their integration with 3D spatial data (3D-LLMs) has seen rapid progress, offering unprecedented capabilities for understanding and interacting with physical spaces. This survey provides a comprehensive overview of the methodologies enabling LLMs to process, understand, and generate 3D data. Highlighting the unique advantages of LLMs, such as in-context learning, step-by-step reasoning, open-vocabulary capabilities, and extensive world knowledge, we underscore their potential to significantly advance spatial comprehension and interaction within embodied Artificial Intelligence (AI) systems. Our investigation spans various 3D data representations, from point clouds to Neural Radiance Fields (NeRFs). It examines their integration with LLMs for tasks such as 3D scene understanding, captioning, question-answering, and dialogue, as well as LLM-based agents for spatial reasoning, planning, and navigation. The paper also includes a brief review of other methods that integrate 3D and language. The meta-analysis presented in this paper reveals significant progress yet underscores the necessity for novel approaches to harness the full potential of 3D-LLMs. Hence, with this paper, we aim to chart a course for future research that explores and expands the capabilities of 3D-LLMs in understanding and interacting with the complex 3D world. To support this survey, we have established a project page where papers related to our topic are organized and listed: https://github.com/ActiveVisionLab/Awesome-LLM-3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14162v2">FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 6 pages, 2 figures, accepted at CIKM 2025 FinAI Workshop
    </div>
    <details class="paper-abstract">
      We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18279v1">Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted to EMNLP 2025 Findings. Previously titled "Text or Pixels? Evaluating Efficiency and Understanding of LLMs with Visual Text Inputs"
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18277v1">Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The increasing number of data a booking platform such as Booking.com and AirBnB offers make it challenging for interested parties to browse through the available accommodations and analyze reviews in an efficient way. Efforts have been made from the booking platform providers to utilize recommender systems in an effort to enable the user to filter the results by factors such as stars, amenities, cost but most valuable insights can be provided by the unstructured text-based reviews. Going through these reviews one-by-one requires a substantial amount of time to be devoted while a respectable percentage of the reviews won't provide to the user what they are actually looking for. This research publication explores how Large Language Models (LLMs) can enhance short rental apartments recommendations by summarizing and mining key insights from user reviews. The web application presented in this paper, named "instaGuide", automates the procedure of isolating the text-based user reviews from a property on the Booking.com platform, synthesizing the summary of the reviews, and enabling the user to query specific aspects of the property in an effort to gain feedback on their personal questions/criteria. During the development of the instaGuide tool, numerous LLM models were evaluated based on accuracy, cost, and response quality. The results suggest that the LLM-powered summarization reduces significantly the amount of time the users need to devote on their search for the right short rental apartment, improving the overall decision-making procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09423v2">Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at https://github.com/Badi-Li/GOAL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19856v1">Model Context Contracts - MCP-Enabled Framework to Integrate LLMs With Blockchain Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      In recent years, blockchain has experienced widespread adoption across various industries, becoming integral to numerous enterprise applications. Concurrently, the rise of generative AI and LLMs has transformed human-computer interactions, offering advanced capabilities in understanding and generating human-like text. The introduction of the MCP has further enhanced AI integration by standardizing communication between AI systems and external data sources. Despite these advancements, there is still no standardized method for seamlessly integrating LLM applications and blockchain. To address this concern, we propose "MCC: Model Context Contracts" a novel framework that enables LLMs to interact directly with blockchain smart contracts through MCP-like protocol. This integration allows AI agents to invoke blockchain smart contracts, facilitating more dynamic and context-aware interactions between users and blockchain networks. Essentially, it empowers users to interact with blockchain systems and perform transactions using queries in natural language. Within this proposed architecture, blockchain smart contracts can function as intelligent agents capable of recognizing user input in natural language and executing the corresponding transactions. To ensure that the LLM accurately interprets natural language inputs and maps them to the appropriate MCP functions, the LLM was fine-tuned using a custom dataset comprising user inputs paired with their corresponding MCP server functions. This fine-tuning process significantly improved the platform's performance and accuracy. To validate the effectiveness of MCC, we have developed an end-to-end prototype implemented on the Rahasak blockchain with the fine-tuned Llama-4 LLM. To the best of our knowledge, this research represents the first approach to using the concept of Model Context Protocol to integrate LLMs with blockchain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19850v1">Prompt Decorators: A Declarative and Composable Syntax for Reasoning, Formatting, and Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are central to reasoning, writing, and decision-support workflows, yet users lack consistent control over how they reason and express outputs. Conventional prompt engineering relies on verbose natural-language instructions, limiting reproducibility, modularity, and interpretability. This paper introduces Prompt Decorators, a declarative, composable syntax that governs LLM behavior through compact control tokens such as +++Reasoning, +++Tone(style=formal), and +++Import(topic="Systems Thinking"). Each decorator modifies a behavioral dimension, such as reasoning style, structure, or tone, without changing task content. The framework formalizes twenty core decorators organized into two functional families (Cognitive & Generative and Expressive & Systemic), each further decomposed into subcategories that govern reasoning, interaction, expression, and session-control. It defines a unified syntax, scoping model, and deterministic processing pipeline enabling predictable and auditable behavior composition. By decoupling task intent from execution behavior, Prompt Decorators create a reusable and interpretable interface for prompt design. Illustrative use cases demonstrate improved reasoning transparency, reduced prompt complexity, and standardized model behavior across domains. The paper concludes with implications for interoperability, behavioral consistency, and the development of declarative interfaces for scalable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23804v3">DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders. To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18250v1">ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Data quality plays a critical role in enhancing supervised fine-tuning (SFT) for large language models (LLMs), and token-level data selection has emerged as a promising direction for its fine-grained nature. Despite their strong empirical performance, existing token-level selection methods share two key limitations: (1) requiring training or accessing an additional reference model, and (2) relying solely on loss information for token selection, which cannot well preserve semantically important tokens that are not favored by loss-based metrics. To address these challenges, we propose ssToken, a Self-modulated and Semantic-aware Token Selection approach. ssToken leverages readily accessible history models to compute the per-token loss difference with the current model, which serves as a self-modulated signal that enables the model to adaptively select tokens along its optimization trajectory, rather than relying on excess loss from an offline-trained reference model as in prior works. We further introduce a semantic-aware, attention-based token importance estimation metric, orthogonal to loss-based selection and providing complementary semantic information for more effective filtering. Extensive experiments across different model families and scales demonstrate that both self-modulated selection and semantic-aware selection alone outperform full-data fine-tuning, while their integration--ssToken--achieves synergistic gains and further surpasses prior token-level selection methods, delivering performance improvements while maintaining training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18245v1">Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 27 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Scaling the number of parameters and the size of training data has proven to be an effective strategy for improving large language model (LLM) performance. Yet, as these models grow increasingly powerful and widely deployed, the cost of inference has become a pressing concern. Despite its importance, the trade-off between model accuracy and inference efficiency remains underexplored. In this work, we examine how key architectural factors, hidden size, the allocation of parameters between MLP and attention (mlp-to-attention ratio), and grouped-query attention (GQA), influence both inference cost and accuracy. We introduce a conditional scaling law that augments the Chinchilla framework with architectural information, along with a search framework for identifying architectures that are simultaneously inference-efficient and accurate. To validate our approach, we train more than 200 models spanning 80M to 3B parameters and 8B to 100B training tokens, and fit the proposed conditional scaling law. Our results show that the conditional scaling law reliably predicts optimal architectural choices and that the resulting models outperform existing open-source baselines. Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16456v2">GPO: Learning from Critical Steps to Improve LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18228v1">Towards Fast LLM Fine-tuning through Zeroth-Order Optimization with Projected Gradient-Aligned Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using zeroth-order (ZO) optimization has emerged as a promising alternative to traditional gradient-based methods due to its reduced memory footprint requirement. However, existing ZO methods suffer from high variance in gradient estimation, leading to slow convergence and suboptimal performance on large-scale models. In this work, we propose P-GAP, a fast LLM fine-tuning approach through zeroth-order optimization with Projected Gradient-Aligned Perturbations. Specifically, we first estimate a low-dimensional gradient space and then align perturbations in projected gradients' direction within the space. This approach enables reduced the number of perturbed parameters and decreased variance, therefore accelerated convergence for LLM fine-tuning. Experiments on LLMs show that P-GAP consistently surpasses the baselines, achieving up to 6% increase in accuracy on classification tasks and up to 12% higher accuracy on generation tasks, with up to about 81% less training iterations and 70% less GPU hours. These results demonstrate that P-GAP enables fast, scalable, and resource-efficient ZO LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14253v2">Towards Agentic Self-Learning LLMs in Search Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://github.com/forangel2014/Towards-Agentic-Self-Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v4">Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) yet often fails to maintain comparable performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Although global pruning aims to identify an optimal sparse model, intuitive methods typically adopt a two-stage paradigm that first evaluates substructure saliency and then applies global pruning, which ignores inter-structure dependencies and fails to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters. Code will be available at https://github.com/AMD-AGI/Tyr-the-Pruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23967v2">HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on Chain-of-Thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off). Specifically, HiPO combines a hybrid data pipelineproviding paired Think-on and Think-off responseswith a hybrid reinforcement learning reward system that balances accuracy and efficiency while avoiding over-reliance on detailed reasoning. Experiments across mathematics and coding benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Finally, we hope HiPO a can be a principled approach for efficient adaptive reasoning, advancing the deployment of reasoning-oriented LLMs in real-world, resource-sensitive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18196v1">Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. We first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges, preventing the search for optimal score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.3% relative improvement on average in Spearman correlation with human judgments across different score ranges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21837v2">Semantic Agreement Enables Efficient Open-Ended LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Industry Track
    </div>
    <details class="paper-abstract">
      Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v6">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18179v1">Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: https://github.com/AdCo-Research/adaptive-coopetition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v3">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11857v3">LLMBridge: Reducing Costs to Access LLMs in a Prompt-Centric Internet</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Today's Internet infrastructure is centered around content retrieval over HTTP, with middleboxes (e.g., HTTP proxies) playing a crucial role in performance, security, and cost-effectiveness. We envision a future where Internet communication will be dominated by "prompts" sent to generative AI models. For this, we will need proxies that provide similar functions to HTTP proxies (e.g., caching, routing, compression) while dealing with unique challenges and opportunities of prompt-based communication. As a first step toward supporting prompt-based communication, we present LLMBridge, an LLM proxy designed for cost-conscious users, such as those in developing regions and education (e.g., students, instructors). LLMBridge supports three key optimizations: model selection (routing prompts to the most suitable model), context management (intelligently reducing the amount of context), and semantic caching (serving prompts using local models and vector databases). These optimizations introduce trade-offs between cost and quality, which applications navigate through a high-level, bidirectional interface. As case studies, we deploy LLMBridge in two cost-sensitive settings: a WhatsApp-based Q&A service and a university classroom environment. The WhatsApp service has been live for over twelve months, serving 100+ users and handling more than 14.7K requests. In parallel, we exposed LLMBridge to students across three computer science courses over a semester, where it supported diverse LLM-powered applications - such as reasoning agents and chatbots - and handled an average of 500 requests per day. We report on deployment experiences across both settings and use the collected workloads to benchmark the effectiveness of various cost-optimization strategies, analyzing their trade-offs in cost, latency, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19107v1">When Your AI Agent Succumbs to Peer-Pressure: Studying Opinion-Change Dynamics of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We investigate how peer pressure influences the opinions of Large Language Model (LLM) agents across a spectrum of cognitive commitments by embedding them in social networks where they update opinions based on peer perspectives. Our findings reveal key departures from traditional conformity assumptions. First, agents follow a sigmoid curve: stable at low pressure, shifting sharply at threshold, and saturating at high. Second, conformity thresholds vary by model: Gemini 1.5 Flash requires over 70% peer disagreement to flip, whereas ChatGPT-4o-mini shifts with a dissenting minority. Third, we uncover a fundamental "persuasion asymmetry," where shifting an opinion from affirmative-to-negative requires a different cognitive effort than the reverse. This asymmetry results in a "dual cognitive hierarchy": the stability of cognitive constructs inverts based on the direction of persuasion. For instance, affirmatively-held core values are robust against opposition but easily adopted from a negative stance, a pattern that inverts for other constructs like attitudes. These dynamics echoing complex human biases like negativity bias, prove robust across different topics and discursive frames (moral, economic, sociotropic). This research introduces a novel framework for auditing the emergent socio-cognitive behaviors of multi-agent AI systems, demonstrating their decision-making is governed by a fluid, context-dependent architecture, not a static logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19099v1">What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 8 pages (main text) + 4 pages (appendix), 4 figures
    </div>
    <details class="paper-abstract">
      Curriculum learning (CL) - ordering training data from easy to hard - has become a popular strategy for improving reasoning in large language models (LLMs). Yet prior work employs disparate difficulty metrics and training setups, leaving open fundamental questions: When does curriculum help? Which direction - forward or reverse - is better? And does the answer depend on what we measure? We address these questions through a unified offline evaluation framework that decomposes curriculum difficulty into five complementary dimensions: Problem Difficulty, Model Surprisal, Confidence Margin, Predictive Uncertainty, and Decision Variability. Through controlled post-training experiments on mathematical reasoning benchmarks with Llama3.1-8B, Mistral-7B, and Gemma3-4B, we find that (i) no curriculum strategy dominates universally - the relative effectiveness of forward versus reverse CL depends jointly on model capability and task complexity; (ii) even within a single metric, samples at different difficulty levels produce distinct gains depending on task demands; and (iii) task-aligned curricula focus on shaping the model's final representations and generalization, whereas inner-state curricula modulate internal states such as confidence and uncertainty. Our findings challenge the notion of a universal curriculum strategy and offer actionable guidance across model and task regimes, with some metrics indicating that prioritizing decision-uncertain samples can further enhance learning outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19060v1">PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 24 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
    </div>
    <details class="paper-abstract">
      While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03502v2">ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 47 pages, 15 figures. Dataset available at Zenodo: https://doi.org/10.5281/zenodo.17249602 Codebase available at GitHub: https://github.com/alikhairallah/ALHD-Benchmarking
    </div>
    <details class="paper-abstract">
      We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19032v1">When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: https://github.com/abeerbadawi/MentalBench/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19028v1">Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19025v1">FlexiDataGen: An Adaptive LLM Framework for Dynamic Semantic Dataset Generation in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Dataset availability and quality remain critical challenges in machine learning, especially in domains where data are scarce, expensive to acquire, or constrained by privacy regulations. Fields such as healthcare, biomedical research, and cybersecurity frequently encounter high data acquisition costs, limited access to annotated data, and the rarity or sensitivity of key events. These issues-collectively referred to as the dataset challenge-hinder the development of accurate and generalizable machine learning models in such high-stakes domains. To address this, we introduce FlexiDataGen, an adaptive large language model (LLM) framework designed for dynamic semantic dataset generation in sensitive domains. FlexiDataGen autonomously synthesizes rich, semantically coherent, and linguistically diverse datasets tailored to specialized fields. The framework integrates four core components: (1) syntactic-semantic analysis, (2) retrieval-augmented generation, (3) dynamic element injection, and (4) iterative paraphrasing with semantic validation. Together, these components ensure the generation of high-quality, domain-relevant data. Experimental results show that FlexiDataGen effectively alleviates data shortages and annotation bottlenecks, enabling scalable and accurate machine learning model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19006v1">XGen-Q: An Explainable Domain-Adaptive LLM Framework with Retrieval-Augmented Generation for Software Security</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Generative AI and large language models (LLMs) have shown strong capabilities in code understanding, but their use in cybersecurity, particularly for malware detection and analysis, remains limited. Existing detection systems often fail to generalize to obfuscated or previously unseen threats, underscoring the need for more adaptable and explainable models. To address this challenge, we introduce XGen-Q, a domain-adapted LLM built on the Qwen-Coder architecture and pretrained on a large-scale corpus of over one million malware samples, spanning both source and assembly code. XGen-Q uses a multi-stage prompt strategy combined with retrieval-augmented generation (RAG) to deliver reliable malware identification and detailed forensic reporting, even in the presence of complex code obfuscation. To further enhance generalization, we design a training pipeline that systematically exposes the model to diverse obfuscation patterns. Experimental results show that XGen-Q achieves significantly lower perplexity than competitive baselines and exhibits strong performance on novel malware samples, demonstrating the promise of LLM-based approaches for interpretable and robust malware analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19005v1">Dynamic Evaluation for Oversensitivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 EMNLP-Findings 2025
    </div>
    <details class="paper-abstract">
      Oversensitivity occurs when language models defensively reject prompts that are actually benign. This behavior not only disrupts user interactions but also obscures the boundary between harmful and harmless content. Existing benchmarks rely on static datasets that degrade overtime as models evolve, leading to data contamination and diminished evaluative power. To address this, we develop a framework that dynamically generates model-specific challenging datasets, capturing emerging defensive patterns and aligning with each model's unique behavior. Building on this approach, we construct OVERBENCH, a benchmark that aggregates these datasets across diverse LLM families, encompassing 450,000 samples from 25 models. OVERBENCH provides a dynamic and evolving perspective on oversensitivity, allowing for continuous monitoring of defensive triggers as models advance, highlighting vulnerabilities that static datasets overlook.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16916v2">SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer promising capabilities for tackling complex reasoning tasks, including optimization problems. However, existing methods either rely on prompt engineering, which leads to poor generalization across problem types, or require costly supervised training. We introduce SolverLLM, a training-free framework that leverages test-time scaling to solve diverse optimization problems. Rather than solving directly, SolverLLM generates mathematical formulations and translates them into solver-ready code, guided by a novel Monte Carlo Tree Search (MCTS) strategy. To enhance the search process, we modify classical MCTS with (1) dynamic expansion for adaptive formulation generation, (2) prompt backpropagation to guide exploration via outcome-driven feedback, and (3) uncertainty backpropagation to incorporate reward reliability into decision-making. Experiments on six standard benchmark datasets demonstrate that SolverLLM outperforms both prompt-based and learning-based baselines, achieving strong generalization without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18932v1">Evaluating LLM Story Generation through Large-scale Network Analysis of Social Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 This paper has 14 pages and 8 figures. To be presented at the NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Evaluating the creative capabilities of large language models (LLMs) in complex tasks often requires human assessments that are difficult to scale. We introduce a novel, scalable methodology for evaluating LLM story generation by analyzing underlying social structures in narratives as signed character networks. To demonstrate its effectiveness, we conduct a large-scale comparative analysis using networks from over 1,200 stories, generated by four leading LLMs (GPT-4o, GPT-4o mini, Gemini 1.5 Pro, and Gemini 1.5 Flash) and a human-written corpus. Our findings, based on network properties like density, clustering, and signed edge weights, show that LLM-generated stories consistently exhibit a strong bias toward tightly-knit, positive relationships, which aligns with findings from prior research using human assessment. Our proposed approach provides a valuable tool for evaluating limitations and tendencies in the creative storytelling of current and future LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18931v1">A Justice Lens on Fairness and Ethics Courses in Computing Education: LLM-Assisted Multi-Perspective and Thematic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 14 pages, 8 figures, In Review
    </div>
    <details class="paper-abstract">
      Course syllabi set the tone and expectations for courses, shaping the learning experience for both students and instructors. In computing courses, especially those addressing fairness and ethics in artificial intelligence (AI), machine learning (ML), and algorithmic design, it is imperative that we understand how approaches to navigating barriers to fair outcomes are being addressed.These expectations should be inclusive, transparent, and grounded in promoting critical thinking. Syllabus analysis offers a way to evaluate the coverage, depth, practices, and expectations within a course. Manual syllabus evaluation, however, is time-consuming and prone to inconsistency. To address this, we developed a justice-oriented scoring rubric and asked a large language model (LLM) to review syllabi through a multi-perspective role simulation. Using this rubric, we evaluated 24 syllabi from four perspectives: instructor, departmental chair, institutional reviewer, and external evaluator. We also prompted the LLM to identify thematic trends across the courses. Findings show that multiperspective evaluation aids us in noting nuanced, role-specific priorities, leveraging them to fill hidden gaps in curricula design of AI/ML and related computing courses focused on fairness and ethics. These insights offer concrete directions for improving the design and delivery of fairness, ethics, and justice content in such courses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18927v1">BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18339v1">ECG-LLM -- training and evaluation of domain-specific large language models for electrocardiography</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 34 pages, 8 figures, code available at https://github.com/AI4HealthUOL/ecg-llm
    </div>
    <details class="paper-abstract">
      Domain-adapted open-weight large language models (LLMs) offer promising healthcare applications, from queryable knowledge bases to multimodal assistants, with the crucial advantage of local deployment for privacy preservation. However, optimal adaptation strategies, evaluation methodologies, and performance relative to general-purpose LLMs remain poorly characterized. We investigated these questions in electrocardiography, an important area of cardiovascular medicine, by finetuning open-weight models on domain-specific literature and implementing a multi-layered evaluation framework comparing finetuned models, retrieval-augmented generation (RAG), and Claude Sonnet 3.7 as a representative general-purpose model. Finetuned Llama 3.1 70B achieved superior performance on multiple-choice evaluations and automatic text metrics, ranking second to Claude 3.7 in LLM-as-a-judge assessments. Human expert evaluation favored Claude 3.7 and RAG approaches for complex queries. Finetuned models significantly outperformed their base counterparts across nearly all evaluation modes. Our findings reveal substantial performance heterogeneity across evaluation methodologies, underscoring assessment complexity. Nevertheless, domain-specific adaptation through finetuning and RAG achieves competitive performance with proprietary models, supporting the viability of privacy-preserving, locally deployable clinical solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18871v1">How Do LLMs Use Their Depth?</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model early on due to the lack of appropriate contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. Even high-frequency token predictions from early layers get refined >70% of the time, indicating that correct token prediction is not "one-and-done". We then go beyond frequency-based prediction to examine the dynamic usage of layer depth across three case studies. (i) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. (ii) Fact recall task analysis shows that, in a multi-token answer, the first token requires more computational depth than the rest. (iii) Multiple-choice task analysis shows that the model identifies the format of the response within the first half of the layers, but finalizes its response only toward the end. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14456v2">Correct-Detect: Balancing Performance and Ambiguity Through the Lens of Coreference Resolution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted at EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intended to reflect human linguistic competencies. But humans have access to a broad and embodied context, which is key in detecting and resolving linguistic ambiguities, even in isolated text spans. A foundational case of semantic ambiguity is found in the task of coreference resolution: how is a pronoun related to an earlier person mention? This capability is implicit in nearly every downstream task, and the presence of ambiguity at this level can alter performance significantly. We show that LLMs can achieve good performance with minimal prompting in both coreference disambiguation and the detection of ambiguity in coreference, however, they cannot do both at the same time. We present the CORRECT-DETECT trade-off: though models have both capabilities and deploy them implicitly, successful performance balancing these two abilities remains elusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03417v2">NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 This paper has been accepted in the main conference proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15926v2">TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      With the emergence of wearable devices and other embedded systems, deploying large language models (LLMs) on edge platforms has become an urgent need. However, this is challenging because of their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as low as 1.58~bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected long latency of the prefill stage. We present \textbf{TeLLMe}, the first table-lookup-based ternary LLM accelerator for low-power edge FPGAs that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. TeLLMe incorporates several novel techniques, including (1) a table-lookup-based ternary matrix multiplication (TLMM) engine utilizing grouped activations and online precomputation for low resource utilization and high throughput; (2) a fine-grained analytic URAM-based weight buffer management scheme for efficient loading and compute engine access; (3) a streaming dataflow architecture that fuses floating-point element-wise operations with linear computations to hide latency; (4) a reversed-reordered prefill stage attention with fused attention operations for high memory efficiency; and (5) a resource-efficient specialized decoding stage attention. Under a 5~W power budget, TeLLMe delivers up to 25~tokens/s decoding throughput and 0.45--0.96~s time-to-first-token (TTFT) for 64--128 token prompts, marking a significant energy-efficiency advancement in LLM inference on edge FPGAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18814v1">Online SFT for LLM Reasoning: Surprising Effectiveness of Self-Tuning without Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      We present a simple, self-help online supervised finetuning (OSFT) paradigm for LLM reasoning. In this paradigm, the model generates its own responses and is immediately finetuned on this self-generated data. OSFT is a highly efficient training strategy for LLM reasoning, as it is reward-free and uses just one rollout by default. Experiment results show that OSFT achieves downstream performance on challenging mathematical reasoning tasks comparable to strong reinforcement learning with verifiable rewards (RLVR) methods such as GRPO. Our ablation study further demonstrates the efficiency and robustness of OSFT. The major mechanism of OSFT lies in facilitating the model's own existing preference (latent knowledge) learned from pretraining, which leads to reasoning ability improvement. We believe that OSFT offers an efficient and promising alternative to more complex, reward-based training paradigms. Our code is available at https://github.com/ElementQi/OnlineSFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v2">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14008v2">Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Updated manuscript of our previous version (arXiv:2502.01714). Under review
    </div>
    <details class="paper-abstract">
      LLM-powered Multi-Agent Systems (LLM-MAS) unlock new potentials in distributed reasoning, collaboration, and task generalization but also introduce additional risks due to unguaranteed agreement, cascading uncertainty, and adversarial vulnerabilities. We argue that ensuring responsible behavior in such systems requires a paradigm shift: from local, superficial agent-level alignment to global, systemic agreement. We conceptualize responsibility not as a static constraint but as a lifecycle-wide property encompassing agreement, uncertainty, and security, each requiring the complementary integration of subjective human-centered values and objective verifiability. Furthermore, a dual-perspective governance framework that combines interdisciplinary design with human-AI collaborative oversight is essential for tracing and ensuring responsibility throughout the lifecycle of LLM-MAS. Our position views LLM-MAS not as loose collections of agents, but as unified, dynamic socio-technical systems that demand principled mechanisms to support each dimension of responsibility and enable ethically aligned, verifiably coherent, and resilient behavior for sustained, system-wide agreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02186v2">EvalAssist: A Human-Centered Tool for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 arXiv admin note: substantial text overlap with arXiv:2410.00873
    </div>
    <details class="paper-abstract">
      With the broad availability of large language models and their ability to generate vast outputs using varied prompts and configurations, determining the best output for a given task requires an intensive evaluation process, one where machine learning practitioners must decide how to assess the outputs and then carefully carry out the evaluation. This process is both time-consuming and costly. As practitioners work with an increasing number of models, they must now evaluate outputs to determine which model and prompt performs best for a given task. LLMs are increasingly used as evaluators to filter training data, evaluate model performance, assess harms and risks, or assist human evaluators with detailed assessments. We present EvalAssist, a framework that simplifies the LLM-as-a-judge workflow. The system provides an online criteria development environment, where users can interactively build, test, and share custom evaluation criteria in a structured and portable format. We support a set of LLM-based evaluation pipelines that leverage off-the-shelf LLMs and use a prompt-chaining approach we developed and contributed to the UNITXT open-source library. Additionally, our system also includes specially trained evaluators to detect harms and risks in LLM outputs. We have deployed the system internally in our organization with several hundreds of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18795v1">ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 17 pages, 5 fiugres
    </div>
    <details class="paper-abstract">
      The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18787v1">ShaRE your Data! Characterizing Datasets for LLM-based Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      [Context] Large Language Models (LLMs) rely on domain-specific datasets to achieve robust performance across training and inference stages. However, in Requirements Engineering (RE), data scarcity remains a persistent limitation reported in surveys and mapping studies. [Question/Problem] Although there are multiple datasets supporting LLM-based RE tasks (LLM4RE), they are fragmented and poorly characterized, limiting reuse and comparability. This research addresses the limited visibility and characterization of datasets used in LLM4RE. We investigate which public datasets are employed, how they can be systematically characterized, and which RE tasks and dataset descriptors remain under-represented. [Ideas/Results] To address this, we conduct a systematic mapping study to identify and analyse datasets used in LLM4RE research. A total of 62 publicly available datasets are referenced across 43 primary studies. Each dataset is characterized along descriptors such as artifact type, granularity, RE stage, task, domain, and language. Preliminary findings show multiple research gaps, including limited coverage for elicitation tasks, scarce datasets for management activities beyond traceability, and limited multilingual availability. [Contribution] This research preview offers a public catalogue and structured characterization scheme to support dataset selection, comparison, and reuse in LLM4RE research. Future work will extend the scope to grey literature, as well as integration with open dataset and benchmark repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10806v2">Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Waiting for Conference Response
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are adept at generating responses based on information within their context. While this ability is useful for interacting with structured data like code files, another popular method, Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment the model's in-context learning. However, it is not well-explored how to best represent this retrieved knowledge for generating responses on structured data, particularly hierarchical structures like trees. In this work, we propose a novel bottom-up method to linearize knowledge from tree-like structures (like a GitHub repository) by generating implicit, aggregated summaries at each hierarchical level. This approach enables the knowledge to be stored in a knowledge base and used directly with RAG. We then compare our method to using RAG on raw, unstructured code, evaluating the accuracy and quality of the generated responses. Our results show that while response quality is comparable across both methods, our approach generates over 68% fewer documents in the retriever, a significant gain in efficiency. This finding suggests that leveraging implicit, linearized knowledge may be a highly effective and scalable strategy for handling complex, hierarchical data structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03655v2">Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation necessitates scalable, automated fact-checking solutions. Yet, current benchmarks often overlook multilingual and topical diversity. This paper introduces a novel, dynamically extensible data set that includes 61,514 claims in multiple languages and topics, extending existing datasets up to 2024. Through a comprehensive evaluation of five prominent Large Language Models (LLMs), including GPT-4o, GPT-3.5 Turbo, LLaMA 3.1, and Mixtral 8x7B, we identify significant performance gaps between different languages and topics. While overall GPT-4o achieves the highest accuracy, it declines to classify 43% of claims. Across all models, factual-sounding claims are misclassified more often than opinions, revealing a key vulnerability. These findings underscore the need for caution and highlight challenges in deploying LLM-based fact-checking systems at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13982v3">Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Preprint; feedback welcome
    </div>
    <details class="paper-abstract">
      What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16551v2">From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      This research proposes a systematic, large language model (LLM) approach for extracting product and service attributes, features, and associated sentiments from customer reviews. Grounded in marketing theory, the framework distinguishes perceptual attributes from actionable features, producing interpretable and managerially actionable insights. We apply the methodology to 20,000 Yelp reviews of Starbucks stores and evaluate eight prompt variants on a random subset of reviews. Model performance is assessed through agreement with human annotations and predictive validity for customer ratings. Results show high consistency between LLMs and human coders and strong predictive validity, confirming the reliability of the approach. Human coders required a median of six minutes per review, whereas the LLM processed each in two seconds, delivering comparable insights at a scale unattainable through manual coding. Managerially, the analysis identifies attributes and features that most strongly influence customer satisfaction and their associated sentiments, enabling firms to pinpoint "joy points," address "pain points," and design targeted interventions. We demonstrate how structured review data can power an actionable marketing dashboard that tracks sentiment over time and across stores, benchmarks performance, and highlights high-leverage features for improvement. Simulations indicate that enhancing sentiment for key service features could yield 1-2% average revenue gains per store.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04710v2">Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      No-Code Development Platforms (NCDPs) empower non-technical end users to build applications tailored to their specific demands without writing code. While NCDPs lower technical barriers, users still require some technical knowledge, e.g., to structure process steps or define event-action rules. Large Language Models (LLMs) offer a promising solution to further reduce technical requirements by supporting natural language interaction and dynamic code generation. By integrating LLM, NCDPs can be more accessible to non-technical users, enabling application development truly without requiring any technical expertise. Despite growing interest in LLM-powered NCDPs, a systematic investigation into the factors influencing LLM suitability and performance remains absent. Understanding these factors is critical to effectively leveraging LLMs capabilities and maximizing their impact. In this paper, we investigate key factors influencing the effectiveness of LLMs in supporting end-user application development within NCDPs. By conducting comprehensive experiments, we evaluate the impact of four key factors, i.e., model selection, prompt language, training data background, and an error-informed few-shot setup, on the quality of generated applications. Specifically, we selected a range of LLMs based on their architecture, scale, design focus, and training data, and evaluated them across four real-world smart home automation scenarios implemented on a representative open-source LLM-powered NCDP. Our findings offer practical insights into how LLMs can be effectively integrated into NCDPs, informing both platform design and the selection of suitable LLMs for end-user application development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18691v1">Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      This study is the first to investigate LLM comprehension capabilities over long-context (LC) medical QA of clinical relevance. Our comprehensive assessment spans a range of content-inclusion settings based on their relevance, LLM models of varying capabilities and datasets across task formulations, revealing insights on model size effects, limitations, underlying memorization issues and the benefits of reasoning models. Importantly, we examine the effect of RAG on medical LC comprehension, uncover best settings in single versus multi-document reasoning datasets and showcase RAG strategies for improvements over LC. We shed light into some of the evaluation aspects using a multi-faceted approach. Our qualitative and error analyses address open questions on when RAG is beneficial over LC, revealing common failure cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17575v2">DeTAILS: Deep Thematic Analysis with Iterative LLM Support</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Thematic analysis is widely used in qualitative research but can be difficult to scale because of its iterative, interpretive demands. We introduce DeTAILS, a toolkit that integrates large language model (LLM) assistance into a workflow inspired by Braun and Clarke's thematic analysis framework. DeTAILS supports researchers in generating and refining codes, reviewing clusters, and synthesizing themes through interactive feedback loops designed to preserve analytic agency. We evaluated the system with 18 qualitative researchers analyzing Reddit data. Quantitative results showed strong alignment between LLM-supported outputs and participants' refinements, alongside reduced workload and high perceived usefulness. Qualitatively, participants reported that DeTAILS accelerated analysis, prompted reflexive engagement with AI outputs, and fostered trust through transparency and control. We contribute: (1) an interactive human-LLM workflow for large-scale qualitative analysis, (2) empirical evidence of its feasibility and researcher experience, and (3) design implications for trustworthy AI-assisted qualitative research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02672v3">EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Accepted by NeurIPS 2025. 47 pages, 24 figures
    </div>
    <details class="paper-abstract">
      We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15732v3">Can LLMs Reconcile Knowledge Conflicts in Counterfactual Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 ICML 2025 Workshop on Scaling up Intervention Models
    </div>
    <details class="paper-abstract">
      Large Language Models have been shown to contain extensive world knowledge in their parameters, enabling impressive performance on many knowledge intensive tasks. However, when deployed in novel settings, LLMs often encounter situations where they must integrate parametric knowledge with new or unfamiliar information. In this work, we explore whether LLMs can combine knowledge in-context with their parametric knowledge through the lens of counterfactual reasoning. Through synthetic and real experiments in multi-hop reasoning problems, we show that LLMs generally struggle with counterfactual reasoning, often resorting to exclusively using their parametric knowledge. Moreover, we show that simple post-hoc finetuning can struggle to instill counterfactual reasoning ability -- often leading to degradation in stored parametric knowledge. Ultimately, our work reveals important limitations of current LLM's abilities to re-purpose parametric knowledge in novel settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18586v1">Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in complex multi-agent applications that use external function calls. This workload creates severe performance challenges for the KV Cache: space contention leads to the eviction of critical agents' caches and time underutilization leaves the cache of agents stalled on long-running tool calls idling in GPU memory. We present Tokencake, a KV-Cache-centric serving framework that co-optimizes scheduling and memory management with an agent-aware design. Tokencake's Space Scheduler uses dynamic memory partitioning to shield critical agents from contention, while its Time Scheduler employs a proactive offload and predictive upload mechanism to repurpose GPU memory during function call stalls. Our evaluation on representative multi-agent benchmarks shows that Tokencake can reduce end-to-end latency by over 47.06%, improve effective GPU memory utilization by up to 16.9% compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.15594v6">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Project Page: https://awesome-llm-as-a-judge.github.io/
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
