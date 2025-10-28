# llm - 2025_10

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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- Part 17

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01428v1">BioVERSE: Representation Alignment of Biomedical Modalities to LLMs for Multi-Modal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and biomedical foundation models (BioFMs) have achieved strong results in biological text reasoning, molecular modeling, and single-cell analysis, yet they remain siloed in disjoint embedding spaces, limiting cross-modal reasoning. We present BIOVERSE (Biomedical Vector Embedding Realignment for Semantic Engagement), a two-stage approach that adapts pretrained BioFMs as modality encoders and aligns them with LLMs through lightweight, modality-specific projection layers. The approach first aligns each modality to a shared LLM space through independently trained projections, allowing them to interoperate naturally, and then applies standard instruction tuning with multi-modal data to bring them together for downstream reasoning. By unifying raw biomedical data with knowledge embedded in LLMs, the approach enables zero-shot annotation, cross-modal question answering, and interactive, explainable dialogue. Across tasks spanning cell-type annotation, molecular description, and protein function reasoning, compact BIOVERSE configurations surpass larger LLM baselines while enabling richer, generative outputs than existing BioFMs, establishing a foundation for principled multi-modal biomedical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01427v1">A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01379v1">Beyond Single LLMs: Enhanced Code Generation via Multi-Stage Performance-Guided LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have become the predominant paradigm for automated code generation, current single-model approaches fundamentally ignore the heterogeneous computational strengths that different models exhibit across programming languages, algorithmic domains, and development stages. This paper challenges the single-model convention by introducing a multi-stage, performance-guided orchestration framework that dynamically routes coding tasks to the most suitable LLMs within a structured generate-fix-refine workflow. Our approach is grounded in a comprehensive empirical study of 17 state-of-the-art LLMs across five programming languages (Python, Java, C++, Go, and Rust) using HumanEval-X benchmark. The study, which evaluates both functional correctness and runtime performance metrics (execution time, mean/max memory utilization, and CPU efficiency), reveals pronounced performance heterogeneity by language, development stage, and problem category. Guided by these empirical insights, we present PerfOrch, an LLM agent that orchestrates top-performing LLMs for each task context through stage-wise validation and rollback mechanisms. Without requiring model fine-tuning, PerfOrch achieves substantial improvements over strong single-model baselines: average correctness rates of 96.22% and 91.37% on HumanEval-X and EffiBench-X respectively, surpassing GPT-4o's 78.66% and 49.11%. Beyond correctness gains, the framework delivers consistent performance optimizations, improving execution time for 58.76% of problems with median speedups ranging from 17.67% to 27.66% across languages on two benchmarks. The framework's plug-and-play architecture ensures practical scalability, allowing new LLMs to be profiled and integrated seamlessly, thereby offering a paradigm for production-grade automated software engineering that adapts to the rapidly evolving generative AI landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01375v1">Fine-tuning with RAG for Improving LLM Learning of New Skills</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Under review at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents deployed for multi-step tasks frequently fail in predictable ways: attempting actions with unmet preconditions, issuing redundant commands, or mishandling environment constraints. While retrieval-augmented generation (RAG) can improve performance by providing runtime guidance, it requires maintaining external knowledge databases and adds computational overhead at every deployment. We propose a simple pipeline that converts inference-time retrieval into learned competence through distillation. Our approach: (1) extracts compact, reusable hints from agent failures, (2) uses these hints to generate improved teacher trajectories via one-shot retrieval at episode start, and (3) trains student models on these trajectories with hint strings removed, forcing internalization rather than memorization. Across two interactive benchmarks, ALFWorld (household tasks) and WebShop (online shopping), distilled students consistently outperform baseline agents, achieving up to 91% success on ALFWorld (vs. 79% for baselines) and improving WebShop scores to 72 (vs. 61 for baselines), while using 10-60% fewer tokens than retrieval-augmented teachers depending on the environment. The approach generalizes across model scales (7B/14B parameters) and agent architectures (ReAct/StateAct), demonstrating that retrieval benefits can be effectively internalized through targeted fine-tuning without permanent runtime dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01363v1">Retrieval-Augmented Framework for LLM-Based Clinical Decision Support</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      The increasing complexity of clinical decision-making, alongside the rapid expansion of electronic health records (EHR), presents both opportunities and challenges for delivering data-informed care. This paper proposes a clinical decision support system powered by Large Language Models (LLMs) to assist prescribing clinicians. The system generates therapeutic suggestions by analyzing historical EHR data, including patient demographics, presenting complaints, clinical symptoms, diagnostic information, and treatment histories. The framework integrates natural language processing with structured clinical inputs to produce contextually relevant recommendations. Rather than replacing clinician judgment, it is designed to augment decision-making by retrieving and synthesizing precedent cases with comparable characteristics, drawing on local datasets or federated sources where applicable. At its core, the system employs a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured narratives and codified data to support LLM-based inference. We outline the system's technical components, including representation representation alignment and generation strategies. Preliminary evaluations, conducted with de-identified and synthetic clinical datasets, examine the clinical plausibility and consistency of the model's outputs. Early findings suggest that LLM-based tools may provide valuable decision support in prescribing workflows when appropriately constrained and rigorously validated. This work represents an initial step toward integration of generative AI into real-world clinical decision-making with an emphasis on transparency, safety, and alignment with established practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23537v2">Beyond the Strongest LLM: Multi-Turn Multi-Agent Orchestration vs. Single LLMs on Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 9 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      We study multi-turn multi-agent orchestration, where multiple large language model (LLM) agents interact over multiple turns by iteratively proposing answers or casting votes until reaching consensus. Using four LLMs (Gemini 2.5 Pro, GPT-5, Grok 4, and Claude Sonnet 4) on GPQA-Diamond, IFEval, and MuSR, we conduct two experiments: (i) benchmarking orchestration against single-LLM baselines; and (ii) ablations on GPQA-Diamond that vary whether agents see who authored answers and whether they can observe ongoing votes. Orchestration matches or exceeds the strongest single model and consistently outperforms the others. Analysis of best-achievable orchestration performance shows potential for further gains. The ablations show that revealing authorship increases self-voting and ties, and that showing ongoing votes amplifies herding, which speeds convergence but can sometimes yield premature consensus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02961v2">Defend LLMs Through Self-Consciousness</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 company requests to withdraw
    </div>
    <details class="paper-abstract">
      This paper introduces a novel self-consciousness defense mechanism for Large Language Models (LLMs) to combat prompt injection attacks. Unlike traditional approaches that rely on external classifiers, our method leverages the LLM's inherent reasoning capabilities to perform self-protection. We propose a framework that incorporates Meta-Cognitive and Arbitration Modules, enabling LLMs to evaluate and regulate their own outputs autonomously. Our approach is evaluated on seven state-of-the-art LLMs using two datasets: AdvBench and Prompt-Injection-Mixed-Techniques-2024. Experiment results demonstrate significant improvements in defense success rates across models and datasets, with some achieving perfect and near-perfect defense in Enhanced Mode. We also analyze the trade-off between defense success rate improvement and computational overhead. This self-consciousness method offers a lightweight, cost-effective solution for enhancing LLM ethics, particularly beneficial for GenAI use cases across various platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01336v1">HiSpec: Hierarchical Speculative Decoding for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by using a smaller draft model to speculate tokens that a larger target model verifies. Verification is often the bottleneck (e.g. verification is $4\times$ slower than token generation when a 3B model speculates for a 70B target model), but most prior works focus only on accelerating drafting. $\textit{``Intermediate"}$ verification reduces verification time by discarding inaccurate draft tokens early, but existing methods incur substantial training overheads in incorporating the intermediate verifier, increase the memory footprint to orchestrate the intermediate verification step, and compromise accuracy by relying on approximate heuristics. We propose $\underline{\textit{Hi}}\textit{erarchical }\underline{\textit{Spec}}\textit{ulative Decoding (HiSpec)}$, a framework for high-throughput speculative decoding that exploits $\textit{early-exit (EE) models}$ for low-overhead intermediate verification. EE models allow tokens to exit early by skipping layer traversal and are explicitly trained so that hidden states at selected layers can be interpreted, making them uniquely suited for intermediate verification without drastically increasing compute and memory overheads. To improve resource-efficiency even further, we design a methodology that enables HiSpec to re-use key-value caches and hidden states between the draft, intermediate verifier, and target models. To maintain accuracy, HiSpec periodically validates the draft tokens accepted by the intermediate verifier against the target model. Our evaluations using various representative benchmarks and models show that HiSpec improves throughput by 1.28$\times$ on average and by up to 2.01$\times$ compared to the baseline single-layer speculation without compromising accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10940v3">CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
      | 💬 Camera-ready
    </div>
    <details class="paper-abstract">
      The full-size MLPs and the projection layers in attention introduce tremendous model sizes of large language models (LLMs), consuming extensive computational resources in pre-training. We empirically observe that the activations of pre-trained LLMs exhibit low-rank property. Motivated by such observations, we propose CoLA and its memory-efficient implementation, CoLA-M, to replace these full-size layers with compute-efficient auto-encoders that naturally enforce low-rank activations throughout training. This fundamental architectural change eliminates the activation redundancy and significantly boosts model capacity and training efficiency. Experiments on LLaMA models with 60 million to 7 billion parameters show that CoLA reduces the computing cost by $\bf 2\pmb{\times}$ and improves training throughput by $\bf 1.86\pmb{\times}$ while maintaining full-rank level performance. CoLA-M further squeezes memory cost without sacrificing throughput, offering a pre-training approach with collectively superior parameter, computing, and memory efficiency. The LLMs produced are also $\bf 2\pmb{\times}$ smaller, enabling faster inference with lower memory cost on resource-constrained platforms.
    </details>
</div>
