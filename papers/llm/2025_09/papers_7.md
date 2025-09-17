# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01822v1">When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has sparked growing interest in their application to time series analysis tasks. However, their ability to perform complex reasoning over temporal data in real-world application domains remains underexplored. To move toward this goal, a first step is to establish a rigorous benchmark dataset for evaluation. In this work, we introduce the TSAIA Benchmark, a first attempt to evaluate LLMs as time-series AI assistants. To ensure both scientific rigor and practical relevance, we surveyed over 20 academic publications and identified 33 real-world task formulations. The benchmark encompasses a broad spectrum of challenges, ranging from constraint-aware forecasting to anomaly detection with threshold calibration: tasks that require compositional reasoning and multi-step time series analysis. The question generator is designed to be dynamic and extensible, supporting continuous expansion as new datasets or task types are introduced. Given the heterogeneous nature of the tasks, we adopt task-specific success criteria and tailored inference-quality metrics to ensure meaningful evaluation for each task. We apply this benchmark to assess eight state-of-the-art LLMs under a unified evaluation protocol. Our analysis reveals limitations in current models' ability to assemble complex time series analysis workflows, underscoring the need for specialized methodologies for domain-specific adaptation. Our benchmark is available at https://huggingface.co/datasets/Melady/TSAIA, and the code is available at https://github.com/USC-Melady/TSAIA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01790v1">Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Prompt sensitivity, referring to the phenomenon where paraphrasing (i.e., repeating something written or spoken using different words) leads to significant changes in large language model (LLM) performance, has been widely accepted as a core limitation of LLMs. In this work, we revisit this issue and ask: Is the widely reported high prompt sensitivity truly an inherent weakness of LLMs, or is it largely an artifact of evaluation processes? To answer this question, we systematically evaluate 7 LLMs (e.g., GPT and Gemini family) across 6 benchmarks, including both multiple-choice and open-ended tasks on 12 diverse prompt templates. We find that much of the prompt sensitivity stems from heuristic evaluation methods, including log-likelihood scoring and rigid answer matching, which often overlook semantically correct responses expressed through alternative phrasings, such as synonyms or paraphrases. When we adopt LLM-as-a-Judge evaluations, we observe a substantial reduction in performance variance and a consistently higher correlation in model rankings across prompts. Our findings suggest that modern LLMs are more robust to prompt templates than previously believed, and that prompt sensitivity may be more an artifact of evaluation than a flaw in the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01750v1">Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Federated learning (FL) for large language models (LLMs) offers a privacy-preserving scheme, enabling clients to collaboratively fine-tune locally deployed LLMs or smaller language models (SLMs) without exchanging raw data. While parameter-sharing methods in traditional FL models solves number of technical challenges, they still incur high communication overhead and struggle with adapting to heterogeneous model architectures. Federated distillation, a framework for mutual knowledge transfer via shared logits, typically offers lower communication overhead than parameter-sharing methods. However, transmitting logits from LLMs remains challenging for bandwidth-limited clients due to their high dimensionality. In this work, we focus on a federated LLM distillation with efficient communication overhead. To achieve this, we first propose an adaptive Top-k logit selection mechanism, dynamically sparsifying logits according to real-time communication conditions. Then to tackle the dimensional inconsistency introduced by the adaptive sparsification, we design an adaptive logits aggregation scheme, effectively alleviating the artificial and uninformative inputs introduced by conventional zero-padding methods. Finally, to enhance the distillation effect, we incorporate LoRA-adapted hidden-layer projection from LLM into the distillation loss, reducing the communication overhead further while providing richer representation. Experimental results demonstrate that our scheme achieves superior performance compared to baseline methods while effectively reducing communication overhead by approximately 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01716v1">An LLM-enabled semantic-centric framework to consume privacy policies</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites, despite claiming otherwise, due to the practical difficulty in comprehending them. The mist of data privacy practices forms a major barrier for user-centred Web approaches, and for data sharing and reusing in an agentic world. Existing research proposed methods for using formal languages and reasoning for verifying the compliance of a specified policy, as a potential cure for ignoring privacy policies. However, a critical gap remains in the creation or acquisition of such formal policies at scale. We present a semantic-centric approach for using state-of-the-art large language models (LLM), to automatically identify key information about privacy practices from privacy policies, and construct $\mathit{Pr}^2\mathit{Graph}$, knowledge graph with grounding from Data Privacy Vocabulary (DPV) for privacy practices, to support downstream tasks. Along with the pipeline, the $\mathit{Pr}^2\mathit{Graph}$ for the top-100 popular websites is also released as a public resource, by using the pipeline for analysis. We also demonstrate how the $\mathit{Pr}^2\mathit{Graph}$ can be used to support downstream tasks by constructing formal policy representations such as Open Digital Right Language (ODRL) or perennial semantic Data Terms of Use (psDToU). To evaluate the technology capability, we enriched the Policy-IE dataset by employing legal experts to create custom annotations. We benchmarked the performance of different large language models for our pipeline and verified their capabilities. Overall, they shed light on the possibility of large-scale analysis of online services' privacy practices, as a promising direction to audit the Web and the Internet. We release all datasets and source code as public resources to facilitate reuse and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01631v1">Unraveling LLM Jailbreaks Through Safety Knowledge Neurons</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01620v1">Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      The rapid development of advanced large language models (LLMs) has made AI-generated text indistinguishable from human-written text. Previous work on detecting AI-generated text has made effective progress, but has not involved modern Chinese poetry. Due to the distinctive characteristics of modern Chinese poetry, it is difficult to identify whether a poem originated from humans or AI. The proliferation of AI-generated modern Chinese poetry has significantly disrupted the poetry ecosystem. Based on the urgency of identifying AI-generated poetry in the real Chinese world, this paper proposes a novel benchmark for detecting LLMs-generated modern Chinese poetry. We first construct a high-quality dataset, which includes both 800 poems written by six professional poets and 41,600 poems generated by four mainstream LLMs. Subsequently, we conduct systematic performance assessments of six detectors on this dataset. Experimental results demonstrate that current detectors cannot be used as reliable tools to detect modern Chinese poems generated by LLMs. The most difficult poetic features to detect are intrinsic qualities, especially style. The detection results verify the effectiveness and necessity of our proposed benchmark. Our work lays a foundation for future detection of AI-generated poetry.
    </details>
</div>
