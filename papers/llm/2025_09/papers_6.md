# llm - 2025_09

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14558v1">LLM Jailbreak Detection for (Almost) Free!</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14543v1">Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into personal writing tools, a critical question arises: can LLMs faithfully imitate an individual's writing style from just a few examples? Personal style is often subtle and implicit, making it difficult to specify through prompts yet essential for user-aligned generation. This work presents a comprehensive evaluation of state-of-the-art LLMs' ability to mimic personal writing styles via in-context learning from a small number of user-authored samples. We introduce an ensemble of complementary metrics-including authorship attribution, authorship verification, style matching, and AI detection-to robustly assess style imitation. Our evaluation spans over 40000 generations per model across domains such as news, email, forums, and blogs, covering writing samples from more than 400 real-world authors. Results show that while LLMs can approximate user styles in structured formats like news and email, they struggle with nuanced, informal writing in blogs and forums. Further analysis on various prompting strategies such as number of demonstrations reveal key limitations in effective personalization. Our findings highlight a fundamental gap in personalized LLM adaptation and the need for improved techniques to support implicit, style-consistent generation. To aid future research and for reproducibility, we open-source our data and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10901v2">3DS: Medical Domain Adaptation of LLMs via Decomposed Difficulty-based Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) excel in general tasks but struggle in specialized domains like healthcare due to limited domain-specific knowledge.Supervised Fine-Tuning(SFT) data construction for domain adaptation often relies on heuristic methods, such as GPT-4 annotation or manual data selection, with a data-centric focus on presumed diverse, high-quality datasets. However, these methods overlook the model's inherent knowledge distribution, introducing noise, redundancy, and irrelevant data, leading to a mismatch between the selected data and the model's learning task, resulting in suboptimal performance. To address this, we propose a two-stage model-centric data selection framework, Decomposed Difficulty Data Selection (3DS), which aligns data with the model's knowledge distribution for optimized adaptation. In Stage1, we apply Prompt-Driven Data Selection via Explicit Alignment, where the the model filters irrelevant or redundant data based on its internal knowledge. In Stage2, we perform Decomposed Difficulty Data Selection, where data selection is guided by our defined difficulty decomposition, using three metrics: Instruction Understanding, Response Confidence, and Response Correctness. Additionally, an attention-based importance weighting mechanism captures token importance for more accurate difficulty calibration. This two-stage approach ensures the selected data is not only aligned with the model's knowledge and preferences but also appropriately challenging for the model to learn, leading to more effective and targeted domain adaptation. In the case study of the medical domain, our extensive experiments on real-world healthcare datasets demonstrate the superiority of 3DS over exisiting methods in accuracy by over 5.29%. Our dataset and code has been open-sourced at https://github.com/PuppyKnightUniversity/3DS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05128v2">DiCoRe: Enhancing Zero-shot Event Detection via Divergent-Convergent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted at EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Zero-shot Event Detection (ED), the task of identifying event mentions in natural language text without any training data, is critical for document understanding in specialized domains. Understanding the complex event ontology, extracting domain-specific triggers from the passage, and structuring them appropriately overloads and limits the utility of Large Language Models (LLMs) for zero-shot ED. To this end, we propose DiCoRe, a divergent-convergent reasoning framework that decouples the task of ED using Dreamer and Grounder. Dreamer encourages divergent reasoning through open-ended event discovery, which helps to boost event coverage. Conversely, Grounder introduces convergent reasoning to align the free-form predictions with the task-specific instructions using finite-state machine guided constrained decoding. Additionally, an LLM-Judge verifies the final outputs to ensure high precision. Through extensive experiments on six datasets across five domains and nine LLMs, we demonstrate how DiCoRe consistently outperforms prior zero-shot, transfer-learning, and reasoning baselines, achieving 4-7% average F1 gains over the best baseline -- establishing DiCoRe as a strong zero-shot ED framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09402v2">Read Before You Think: Mitigating LLM Comprehension Failures with Step-by-Step Reading</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Done in November 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often fail on complex reasoning tasks due to flawed question comprehension, not just flawed logic. This paper presents a systematic investigation into these comprehension failures. Our work yields three key insights: (1) the step-by-step principle, effective for calculation, can be migrated to the reading process to enhance comprehension; (2) increasing the proportion of question-related tokens (e.g., via repetition) succeeds by refocusing attention, a mechanism that can be explicitly controlled; and (3) backward dependencies represent a core bottleneck for decoder-only models that persists even with strong methods like Chain-of-Thought. Based on these findings, we introduce the Step-by-Step Reading (SSR) family of prompts. This multi-stage approach culminates in SSR++, a method specifically engineered to deepen model comprehension by guiding it to parse questions with finer granularity, focus attention on critical tokens, and resolve backward dependencies through iterative re-contextualization. SSR++ sets a new state-of-the-art on multiple reasoning benchmarks, and our analysis confirms it works by directly mitigating semantic misunderstanding. These results demonstrate that guiding how a model reads is a powerful and efficient method for improving its reasoning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04083v2">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 New experiments added with the GPT-OSS-120B model
    </div>
    <details class="paper-abstract">
      Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks. Methods: We use OpenAI GPT-4-turbo and OpenAI's reasoning models o1 and GPT-OSS to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language. Results: Our work is the first to study and compare the performance of the GPT-4, o1 and GPT-OSS for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions. Conclusion: LLMs exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation costs and NLP modeling needs but with increased perpetual compute costs. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available for additional benchmarking by the community: https://github.com/bionlproc/ZeroShotRE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10179v2">Benchmark of stylistic variation in LLM-generated texts</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Data and scripts: https://osf.io/hs7xt/. Interactive charts: https://www.korpus.cz/stylisticbenchmark/
    </div>
    <details class="paper-abstract">
      This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15555v2">Bayesian Concept Bottleneck Models with LLM Priors</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 2025 Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) have been proposed as a compromise between white-box and black-box models, aiming to achieve interpretability without sacrificing accuracy. The standard training procedure for CBMs is to predefine a candidate set of human-interpretable concepts, extract their values from the training data, and identify a sparse subset as inputs to a transparent prediction model. However, such approaches are often hampered by the tradeoff between exploring a sufficiently large set of concepts versus controlling the cost of obtaining concept extractions, resulting in a large interpretability-accuracy tradeoff. This work investigates a novel approach that sidesteps these challenges: BC-LLM iteratively searches over a potentially infinite set of concepts within a Bayesian framework, in which Large Language Models (LLMs) serve as both a concept extraction mechanism and prior. Even though LLMs can be miscalibrated and hallucinate, we prove that BC-LLM can provide rigorous statistical inference and uncertainty quantification. Across image, text, and tabular datasets, BC-LLM outperforms interpretable baselines and even black-box models in certain settings, converges more rapidly towards relevant concepts, and is more robust to out-of-distribution samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15455v1">IMPQ: Interaction-Aware Layerwise Mixed Precision Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise impressive capabilities, yet their multi-billion-parameter scale makes on-device or low-resource deployment prohibitive. Mixed-precision quantization offers a compelling solution, but existing methods struggle when the average precision drops below four bits, as they rely on isolated, layer-specific metrics that overlook critical inter-layer interactions affecting overall performance. In this paper, we propose two innovations to address these limitations. First, we frame the mixed-precision quantization problem as a cooperative game among layers and introduce Shapley-based Progressive Quantization Estimation (SPQE) to efficiently obtain accurate Shapley estimates of layer sensitivities and inter-layer interactions. Second, building upon SPQE, we propose Interaction-aware Mixed-Precision Quantization (IMPQ) which translates these Shapley estimates into a binary quadratic optimization formulation, assigning either 2 or 4-bit precision to layers under strict memory constraints. Comprehensive experiments conducted on Llama-3, Gemma-2, and Qwen-3 models across three independent PTQ backends (Quanto, HQQ, GPTQ) demonstrate IMPQ's scalability and consistently superior performance compared to methods relying solely on isolated metrics. Across average precisions spanning 4 bit down to 2 bit, IMPQ cuts Perplexity by 20 to 80 percent relative to the best baseline, with the margin growing as the bit-width tightens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21772v2">Calibrating LLM Confidence by Probing Perturbed Representation Stability</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v5">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 10 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15474v3">Subjective Behaviors and Preferences in LLM: Language of Browsing</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15335v1">PolBiX: Detecting LLMs' Political Bias in Fact-Checking through X-phemisms</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly used in applications requiring objective assessment, which could be compromised by political bias. Many studies found preferences for left-leaning positions in LLMs, but downstream effects on tasks like fact-checking remain underexplored. In this study, we systematically investigate political bias through exchanging words with euphemisms or dysphemisms in German claims. We construct minimal pairs of factually equivalent claims that differ in political connotation, to assess the consistency of LLMs in classifying them as true or false. We evaluate six LLMs and find that, more than political leaning, the presence of judgmental words significantly influences truthfulness assessment. While a few models show tendencies of political bias, this is not mitigated by explicitly calling for objectivism in prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15283v1">Evaluating the Limitations of Local LLMs in Solving Complex Programming Challenges</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Comments: 16 pages, 3 figures, 8 tables, accepted to CCSC Eastern 2025
    </div>
    <details class="paper-abstract">
      This study examines the performance of today's open-source, locally hosted large-language models (LLMs) in handling complex competitive programming tasks with extended problem descriptions and contexts. Building on the original Framework for AI-driven Code Generation Evaluation (FACE), the authors retrofit the pipeline to work entirely offline through the Ollama runtime, collapsing FACE's sprawling per-problem directory tree into a handful of consolidated JSON files, and adding robust checkpointing so multi-day runs can resume after failures. The enhanced framework generates, submits, and records solutions for the full Kattis corpus of 3,589 problems across eight code-oriented models ranging from 6.7-9 billion parameters. The submission results show that the overall pass@1 accuracy is modest for the local models, with the best models performing at approximately half the acceptance rate of the proprietary models, Gemini 1.5 and ChatGPT-4. These findings expose a persistent gap between private, cost-controlled LLM deployments and state-of-the-art proprietary services, yet also highlight the rapid progress of open models and the practical benefits of an evaluation workflow that organizations can replicate on in-house hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03853v6">Dr. Jekyll and Mr. Hyde: Two Faces of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being integrated into applications such as chatbots or email assistants. To prevent improper responses, safety mechanisms, such as Reinforcement Learning from Human Feedback (RLHF), are implemented in them. In this work, we bypass these safety measures for ChatGPT, Gemini, and Deepseek by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information when querying ChatGPT, Gemini, and Deepseek. We show that these chatbots are vulnerable to this attack by getting dangerous information for 40 out of 40 illicit questions in GPT-4.1-mini, Gemini-1.5-flash, 39 out of 40 in GPT-4o-mini, 38 out of 40 in GPT-3.5-turbo, and 2 out of 2 cases in Gemini-2.5-flash and DeepSeek V3. The attack can be carried out manually or automatically using a support LLM, and has proven effective against models deployed between 2023 and 2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15260v1">Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 9 pages, EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12566v3">Exploring the Impact of Personality Traits on LLM Bias and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16275v1">SecureFixAgent: A Hybrid LLM Agent for Automated Python Static Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 6 pages, 3 figures, 4 tables, 1 algorithm, accepted in the Robustness and Security of Large Language Models (ROSE-LLM) special session at ICMLA 2025
    </div>
    <details class="paper-abstract">
      Modern software development pipelines face growing challenges in securing large codebases with extensive dependencies. Static analysis tools like Bandit are effective at vulnerability detection but suffer from high false positives and lack repair capabilities. Large Language Models (LLMs), in contrast, can suggest fixes but often hallucinate changes and lack self-validation. We present SecureFixAgent, a hybrid repair framework integrating Bandit with lightweight local LLMs (<8B parameters) in an iterative detect-repair-validate loop. To improve precision, we apply parameter-efficient LoRA-based fine-tuning on a diverse, curated dataset spanning multiple Python project domains, mitigating dataset bias and reducing unnecessary edits. SecureFixAgent uses Bandit for detection, the LLM for candidate fixes with explanations, and Bandit re-validation for verification, all executed locally to preserve privacy and reduce cloud reliance. Experiments show SecureFixAgent reduces false positives by 10.8% over static analysis, improves fix accuracy by 13.51%, and lowers false positives by 5.46% compared to pre-trained LLMs, typically converging within three iterations. Beyond metrics, developer studies rate explanation quality 4.5/5, highlighting its value for human trust and adoption. By combining verifiable security improvements with transparent rationale in a resource-efficient local framework, SecureFixAgent advances trustworthy, automated vulnerability remediation for modern pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13869v1">Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 38 pages, 31 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13868v1">Are Prompts All You Need? Evaluating Prompt-Based Large Language Models (LLM)s for Software Requirements Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 33 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Requirements classification assigns natural language requirements to predefined classes, such as functional and non functional. Accurate classification reduces risk and improves software quality. Most existing models rely on supervised learning, which needs large labeled data that are costly, slow to create, and domain dependent; they also generalize poorly and often require retraining for each task. This study tests whether prompt based large language models can reduce data needs. We benchmark several models and prompting styles (zero shot, few shot, persona, and chain of thought) across multiple tasks on two English datasets, PROMISE and SecReq. For each task we compare model prompt configurations and then compare the best LLM setups with a strong fine tuned transformer baseline. Results show that prompt based LLMs, especially with few shot prompts, can match or exceed the baseline. Adding a persona, or persona plus chain of thought, can yield further gains. We conclude that prompt based LLMs are a practical and scalable option that reduces dependence on large annotations and can improve generalizability across tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11176v4">LogiDynamics: Unraveling the Dynamics of Inductive, Abductive and Deductive Logical Inferences in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) employ diverse logical inference mechanisms for reasoning, making the strategic optimization of these approaches critical for advancing their capabilities. This paper systematically investigate the comparative dynamics of inductive (System 1) versus abductive/deductive (System 2) inference in LLMs. We utilize a controlled analogical reasoning environment, varying modality (textual, visual, symbolic), difficulty, and task format (MCQ / free-text). Our analysis reveals System 2 pipelines generally excel, particularly in visual/symbolic modalities and harder tasks, while System 1 is competitive for textual and easier problems. Crucially, task format significantly influences their relative advantage, with System 1 sometimes outperforming System 2 in free-text rule-execution. These core findings generalize to broader in-context learning. Furthermore, we demonstrate that advanced System 2 strategies like hypothesis selection and iterative refinement can substantially scale LLM reasoning. This study offers foundational insights and actionable guidelines for strategically deploying logical inference to enhance LLM reasoning. Resources are available at https://github.com/HKUST-KnowComp/LogiDynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13790v1">Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13765v1">TENET: An Efficient Sparsity-Aware LUT-Centric Architecture for Ternary LLM Inference On Edge</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Ternary quantization has emerged as a powerful technique for reducing both computational and memory footprint of large language models (LLM), enabling efficient real-time inference deployment without significantly compromising model accuracy. Conventional LLM inference platforms (e.g GPUs) cannot capitalize on its benefits, as they (i) lack native support for ternary arithmetic and memory specialization and (ii) remain severely under-utilized in low-batch, real-time scenarios. In this work, we propose TENET, a sparse-aware LUT-centric architecture that co-optimizes algorithm, compute, and memory for ternary LLM inference. To maximize the efficiency of Ternary Linear layer, TENET introduces a Sparse Ternary LUT (STL) core that optimizes ternary mixed-precision GEMM using a symmetric precompute lookup table. It also features Dynamic Activation N:M Sparsity to exploit the sparsity within the activation of each token. Additionally, we propose a LUT-based 64B:80B ternary weight decompression module to fully exploit the memory efficiency of ternary values. At the system level, we design a heterogeneous TENET accelerator with full programmability that integrates STL cores with high-precision cores. An associated Linear-Projection-aware Sparse Attention dataflow is introduced to optimize memory access and hardware utilization. We implement TENET accelerator prototype on both FPGA and ASIC platforms. Experiments across various model sizes and workloads demonstrate that TENET-FPGA and TENET-ASIC improve energy efficiency by 4.3$\times$ and 21.1$\times$, respectively, compared to the A100 GPU. Furthermore, TENET-ASIC achieves a 2.7$\times$ average speedup compared to the A100 GPU in end-to-end inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15495v3">SynthCoder: A Synthetical Strategy to Tune LLMs for Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Code completion is a prominent application of Large Language Models (LLMs) in software engineering. Due to the near real-time response requirements of this task, base models with small to medium-sized parameters are typically employed, supplemented by various optimization and post-training techniques. However, these optimization methods often have trade-offs, leading to a seesaw effect where performance improvements on certain datasets or metrics are accompanied by degradations on others -- sometimes even falling below the baseline model's performance. This paper proposes SynthCoder, a model that integrates leading industry practices to achieve state-of-the-art performance on the Fill-in-the-Middle (FIM) code completion task. In specific, we first construct a diverse dataset by combining Abstract Syntax Tree (AST) node extraction with heuristics that simulate developer behavior. Then we enrich our training corpus with cross-file contextual information using the BM25 algorithm and call graphs, enhancing the model's ability to perform code completion in both file-level and repository-level scenarios. As the last step, we employ a two-stage training process using the Seed-Coder-8B-Base as the base model. First, we fine-tune the model using Curriculum Learning technology. Following this, we perform alignment using Direct Preference Optimization (DPO) with preference pairs generated through Rejection Sampling. Experimental results demonstrate that our final model excels on mainstream repository-level code completion benchmarks, including aiXcoder, ExecRepoBench, CrossCodeEval, and CoLT. Furthermore, our carefully curated training set effectively mitigates the model's tendency to just repeat existing code, a common issue existing in various code completion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19894v4">Semantic Alignment-Enhanced Code Translation via an LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages. In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13710v1">CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized various aspects of human life, yet their immense computational and energy demands pose significant challenges for efficient inference. The memory wall, the growing processor-memory speed disparity, remains a critical bottleneck for LLM. Process-In-Memory (PIM) architectures overcome limitations by co-locating compute units with memory, leveraging 5-20$\times$ higher internal bandwidth and enabling greater energy efficiency than GPUs. However, existing PIMs struggle to balance flexibility, performance, and cost-efficiency for LLMs' dynamic memory-compute patterns and operator diversity. DRAM-PIM suffers from inter-bank communication overhead despite its vector parallelism. SRAM-PIM offers sub-10ns latency for matrix operation but is constrained by limited capacity. This work introduces CompAir, a novel PIM architecture that integrates DRAM-PIM and SRAM-PIM with hybrid bonding, enabling efficient linear computations while unlocking multi-granularity data pathways. We further develop CompAir-NoC, an advanced network-on-chip with an embedded arithmetic logic unit that performs non-linear operations during data movement, simultaneously reducing communication overhead and area cost. Finally, we develop a hierarchical Instruction Set Architecture that ensures both flexibility and programmability of the hybrid PIM. Experimental results demonstrate that CompAir achieves 1.83-7.98$\times$ prefill and 1.95-6.28$\times$ decode improvement over the current state-of-the-art fully PIM architecture. Compared to the hybrid A100 and HBM-PIM system, CompAir achieves 3.52$\times$ energy consumption reduction with comparable throughput. This work represents the first systematic exploration of hybrid DRAM-PIM and SRAM-PIM architectures with in-network computation capabilities, offering a high-efficiency solution for LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07445v2">Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings, indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13694v1">StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Accepted by MICRO'25
    </div>
    <details class="paper-abstract">
      Efficient execution of deep learning workloads on dataflow architectures is crucial for overcoming memory bottlenecks and maximizing performance. While streaming intermediate results between computation kernels can significantly improve efficiency, existing approaches struggle with inter-kernel correlations, external memory access management, and buffer optimization. In this work, we propose StreamTensor, a compiler framework that automatically constructs and optimizes stream-based dataflow accelerators. StreamTensor introduces a novel iterative tensor type system to explicitly encode stream layouts, enabling seamless kernel fusion, buffer allocation, and memory optimization. By systematically exploring three hierarchical design spaces, including tensor tiling, kernel fusion, and resource allocation, StreamTensor balances computational intensity, memory efficiency, and data streaming to maximize performance. Based on FPGA evaluations on Large Language Models (LLM), StreamTensor achieves up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs, making it a promising approach for scalable dataflow-based deep learning acceleration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13691v1">SPAR: Scalable LLM-based PDDL Domain Generation for Aerial Robotics</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We investigate the problem of automatic domain generation for the Planning Domain Definition Language (PDDL) using Large Language Models (LLMs), with a particular focus on unmanned aerial vehicle (UAV) tasks. Although PDDL is a widely adopted standard in robotic planning, manually designing domains for diverse applications such as surveillance, delivery, and inspection is labor-intensive and error-prone, which hinders adoption and real-world deployment. To address these challenges, we propose SPAR, a framework that leverages the generative capabilities of LLMs to automatically produce valid, diverse, and semantically accurate PDDL domains from natural language input. To this end, we first introduce a systematically formulated and validated UAV planning dataset, consisting of ground-truth PDDL domains and associated problems, each paired with detailed domain and action descriptions. Building on this dataset, we design a prompting framework that generates high-quality PDDL domains from language input. The generated domains are evaluated through syntax validation, executability, feasibility, and interpretability. Overall, this work demonstrates that LLMs can substantially accelerate the creation of complex planning domains, providing a reproducible dataset and evaluation pipeline that enables application experts without prior experience to leverage it for practical tasks and advance future research in aerial robotics and automated planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09061v3">EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 4 figures, 7 pages, IEEE conference template
    </div>
    <details class="paper-abstract">
      This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13664v1">Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 To be appeared in EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13642v1">LLM-I: LLMs are Naturally Interleaved Multimodal Creators</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We propose LLM-Interleaved (LLM-I), a flexible and dynamic framework that reframes interleaved image-text generation as a tool-use problem. LLM-I is designed to overcome the "one-tool" bottleneck of current unified models, which are limited to synthetic imagery and struggle with tasks requiring factual grounding or programmatic precision. Our framework empowers a central LLM or MLLM agent to intelligently orchestrate a diverse toolkit of specialized visual tools, including online image search, diffusion-based generation, code execution, and image editing. The agent is trained to select and apply these tools proficiently via a Reinforcement Learning (RL) framework that features a hybrid reward system combining rule-based logic with judgments from LLM and MLLM evaluators. Trained on a diverse new dataset using four different model backbones, LLM-I demonstrates state-of-the-art performance, outperforming existing methods by a large margin across four benchmarks. We also introduce a novel test-time scaling strategy that provides further performance gains. Project Page: https://github.com/ByteDance-BandAI/LLM-I.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13624v1">Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Camera-ready version. Accepted to appear in the proceedings of the 14th Joint Conference on Lexical and Computational Semantics (*SEM 2025)
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed across diverse applications. This often includes tasks LLMs have not encountered during training. This implies that enumerating and obtaining the high-quality training data for all tasks is infeasible. Thus, we often need to rely on transfer learning using datasets with different characteristics, and anticipate out-of-distribution requests. Motivated by this practical need, we propose an analysis framework, building a transfer learning matrix and dimensionality reduction, to dissect these cross-task interactions. We train and analyze 10 models to identify latent abilities (e.g., Reasoning, Sentiment Classification, NLU, Arithmetic) and discover the side effects of the transfer learning. Our findings reveal that performance improvements often defy explanations based on surface-level dataset similarity or source data quality. Instead, hidden statistical factors of the source dataset, such as class distribution and generation length proclivities, alongside specific linguistic features, are actually more influential. This work offers insights into the complex dynamics of transfer learning, paving the way for more predictable and effective LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11963v3">NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 v3: Revisions with added experiments, clarifications, and related work updates
    </div>
    <details class="paper-abstract">
      The capability of large language models to handle long-context information is crucial across various real-world applications. Existing evaluation methods often rely either on real-world long texts, making it difficult to exclude the influence of models' inherent knowledge, or introduce irrelevant filler content to artificially achieve target lengths, reducing assessment effectiveness. To address these limitations, we introduce NeedleBench, a synthetic framework for assessing retrieval and reasoning performance in bilingual long-context tasks with adaptive context lengths. NeedleBench systematically embeds key data points at varying depths to rigorously test model capabilities. Tasks are categorized into two scenarios: information-sparse, featuring minimal relevant details within extensive irrelevant text to simulate simple retrieval tasks; and information-dense (the Ancestral Trace Challenge), where relevant information is continuously distributed throughout the context to simulate complex reasoning tasks. Our experiments reveal that although recent reasoning models like Deepseek-R1 and OpenAI's o3 excel in mathematical reasoning, they struggle with continuous retrieval and reasoning in information-dense scenarios, even at shorter context lengths. We also characterize a phenomenon termed 'under-thinking', where models prematurely conclude reasoning despite available information. NeedleBench thus provides critical insights and targeted tools essential for evaluating and improving LLMs' long-context capabilities. All resources are available at OpenCompass: https://github.com/open-compass/opencompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13603v1">Modernizing Facebook Scoped Search: Keyword and Embedding Hybrid Retrieval with LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 5 Pages, work done as Yongye Su's internship project at Meta
    </div>
    <details class="paper-abstract">
      Beyond general web-scale search, social network search uniquely enables users to retrieve information and discover potential connections within their social context. We introduce a framework of modernized Facebook Group Scoped Search by blending traditional keyword-based retrieval with embedding-based retrieval (EBR) to improve the search relevance and diversity of search results. Our system integrates semantic retrieval into the existing keyword search pipeline, enabling users to discover more contextually relevant group posts. To rigorously assess the impact of this blended approach, we introduce a novel evaluation framework that leverages large language models (LLMs) to perform offline relevance assessments, providing scalable and consistent quality benchmarks. Our results demonstrate that the blended retrieval system significantly enhances user engagement and search quality, as validated by both online metrics and LLM-based evaluation. This work offers practical insights for deploying and evaluating advanced retrieval systems in large-scale, real-world social platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14483v1">An LLM-based multi-agent framework for agile effort estimation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Submitted to ASE'25
    </div>
    <details class="paper-abstract">
      Effort estimation is a crucial activity in agile software development, where teams collaboratively review, discuss, and estimate the effort required to complete user stories in a product backlog. Current practices in agile effort estimation heavily rely on subjective assessments, leading to inaccuracies and inconsistencies in the estimates. While recent machine learning-based methods show promising accuracy, they cannot explain or justify their estimates and lack the capability to interact with human team members. Our paper fills this significant gap by leveraging the powerful capabilities of Large Language Models (LLMs). We propose a novel LLM-based multi-agent framework for agile estimation that not only can produce estimates, but also can coordinate, communicate and discuss with human developers and other agents to reach a consensus. Evaluation results on a real-life dataset show that our approach outperforms state-of-the-art techniques across all evaluation metrics in the majority of the cases. Our human study with software development practitioners also demonstrates an overwhelmingly positive experience in collaborating with our agents in agile effort estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14478v1">Estimating Semantic Alphabet Size for LLM Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Many black-box techniques for quantifying the uncertainty of large language models (LLMs) rely on repeated LLM sampling, which can be computationally expensive. Therefore, practical applicability demands reliable estimation from few samples. Semantic entropy (SE) is a popular sample-based uncertainty estimator with a discrete formulation attractive for the black-box setting. Recent extensions of semantic entropy exhibit improved LLM hallucination detection, but do so with less interpretable methods that admit additional hyperparameters. For this reason, we revisit the canonical discrete semantic entropy estimator, finding that it underestimates the "true" semantic entropy, as expected from theory. We propose a modified semantic alphabet size estimator, and illustrate that using it to adjust discrete semantic entropy for sample coverage results in more accurate semantic entropy estimation in our setting of interest. Furthermore, our proposed alphabet size estimator flags incorrect LLM responses as well or better than recent top-performing approaches, with the added benefit of remaining highly interpretable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14464v1">Not What the Doctor Ordered: Surveying LLM-based De-identification and Quantifying Clinical Information Loss</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      De-identification in the healthcare setting is an application of NLP where automated algorithms are used to remove personally identifying information of patients (and, sometimes, providers). With the recent rise of generative large language models (LLMs), there has been a corresponding rise in the number of papers that apply LLMs to de-identification. Although these approaches often report near-perfect results, significant challenges concerning reproducibility and utility of the research papers persist. This paper identifies three key limitations in the current literature: inconsistent reporting metrics hindering direct comparisons, the inadequacy of traditional classification metrics in capturing errors which LLMs may be more prone to (i.e., altering clinically relevant information), and lack of manual validation of automated metrics which aim to quantify these errors. To address these issues, we first present a survey of LLM-based de-identification research, highlighting the heterogeneity in reporting standards. Second, we evaluated a diverse set of models to quantify the extent of inappropriate removal of clinical information. Next, we conduct a manual validation of an existing evaluation metric to measure the removal of clinical information, employing clinical experts to assess their efficacy. We highlight poor performance and describe the inherent limitations of such metrics in identifying clinically significant changes. Lastly, we propose a novel methodology for the detection of clinically relevant information removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12476v2">Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14456v1">Correct-Detect: Balancing Performance and Ambiguity Through the Lens of Coreference Resolution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intended to reflect human linguistic competencies. But humans have access to a broad and embodied context, which is key in detecting and resolving linguistic ambiguities, even in isolated text spans. A foundational case of semantic ambiguity is found in the task of coreference resolution: how is a pronoun related to an earlier person mention? This capability is implicit in nearly every downstream task, and the presence of ambiguity at this level can alter performance significantly. We show that LLMs can achieve good performance with minimal prompting in both coreference disambiguation and the detection of ambiguity in coreference, however, they cannot do both at the same time. We present the CORRECT-DETECT trade-off: though models have both capabilities and deploy them implicitly, successful performance balancing these two abilities remains elusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14448v1">VCBench: Benchmarking LLMs in Venture Capital</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Benchmarks such as SWE-bench and ARC-AGI demonstrate how shared datasets accelerate progress toward artificial general intelligence (AGI). We introduce VCBench, the first benchmark for predicting founder success in venture capital (VC), a domain where signals are sparse, outcomes are uncertain, and even top investors perform modestly. At inception, the market index achieves a precision of 1.9%. Y Combinator outperforms the index by a factor of 1.7x, while tier-1 firms are 2.9x better. VCBench provides 9,000 anonymized founder profiles, standardized to preserve predictive features while resisting identity leakage, with adversarial tests showing more than 90% reduction in re-identification risk. We evaluate nine state-of-the-art large language models (LLMs). DeepSeek-V3 delivers over six times the baseline precision, GPT-4o achieves the highest F0.5, and most models surpass human benchmarks. Designed as a public and evolving resource available at vcbench.com, VCBench establishes a community-driven standard for reproducible and privacy-preserving evaluation of AGI in early-stage venture forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19477v3">Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14405v1">Adding LLMs to the psycholinguistic norming toolbox: A practical guide to getting the most out of human ratings</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Word-level psycholinguistic norms lend empirical support to theories of language processing. However, obtaining such human-based measures is not always feasible or straightforward. One promising approach is to augment human norming datasets by using Large Language Models (LLMs) to predict these characteristics directly, a practice that is rapidly gaining popularity in psycholinguistics and cognitive science. However, the novelty of this approach (and the relative inscrutability of LLMs) necessitates the adoption of rigorous methodologies that guide researchers through this process, present the range of possible approaches, and clarify limitations that are not immediately apparent, but may, in some cases, render the use of LLMs impractical. In this work, we present a comprehensive methodology for estimating word characteristics with LLMs, enriched with practical advice and lessons learned from our own experience. Our approach covers both the direct use of base LLMs and the fine-tuning of models, an alternative that can yield substantial performance gains in certain scenarios. A major emphasis in the guide is the validation of LLM-generated data with human "gold standard" norms. We also present a software framework that implements our methodology and supports both commercial and open-weight models. We illustrate the proposed approach with a case study on estimating word familiarity in English. Using base models, we achieved a Spearman correlation of 0.8 with human ratings, which increased to 0.9 when employing fine-tuned models. This methodology, framework, and set of best practices aim to serve as a reference for future research on leveraging LLMs for psycholinguistic and lexical studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14404v1">A Taxonomy of Prompt Defects in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become key components of modern software, with prompts acting as their de-facto programming interface. However, prompt design remains largely empirical and small mistakes can cascade into unreliable, insecure, or inefficient behavior. This paper presents the first systematic survey and taxonomy of prompt defects, recurring ways that prompts fail to elicit their intended behavior from LLMs. We organize defects along six dimensions: (1) Specification and Intent, (2) Input and Content, (3) Structure and Formatting, (4) Context and Memory, (5) Performance and Efficiency, and (6) Maintainability and Engineering. Each dimension is refined into fine-grained subtypes, illustrated with concrete examples and root cause analysis. Grounded in software engineering principles, we show how these defects surface in real development workflows and examine their downstream effects. For every subtype, we distill mitigation strategies that span emerging prompt engineering patterns, automated guardrails, testing harnesses, and evaluation frameworks. We then summarize these strategies in a master taxonomy that links defect, impact, and remedy. We conclude with open research challenges and a call for rigorous engineering-oriented methodologies to ensure that LLM-driven systems are dependable by design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14391v1">Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Extending LLM context windows is crucial for long range tasks. RoPE-based position interpolation (PI) methods like linear and frequency-aware scaling extend input lengths without retraining, while post-training quantization (PTQ) enables practical deployment. We show that combining PI with PTQ degrades accuracy due to coupled effects long context aliasing, dynamic range dilation, axis grid anisotropy, and outlier shifting that induce position-dependent logit noise. We provide the first systematic analysis of PI plus PTQ and introduce two diagnostics: Interpolation Pressure (per-band phase scaling sensitivity) and Tail Inflation Ratios (outlier shift from short to long contexts). To address this, we propose Q-ROAR, a RoPE-aware, weight-only stabilization that groups RoPE dimensions into a few frequency bands and performs a small search over per-band scales for W_Q,W_K, with an optional symmetric variant to preserve logit scale. The diagnostics guided search uses a tiny long-context dev set and requires no fine-tuning, kernel, or architecture changes. Empirically, Q-ROAR recovers up to 0.7% accuracy on standard tasks and reduces GovReport perplexity by more than 10%, while preserving short-context performance and compatibility with existing inference stacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14335v1">Beyond Classification: Evaluating LLMs for Fine-Grained Automatic Malware Behavior Auditing</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Automated malware classification has achieved strong detection performance. Yet, malware behavior auditing seeks causal and verifiable explanations of malicious activities -- essential not only to reveal what malware does but also to substantiate such claims with evidence. This task is challenging, as adversarial intent is often hidden within complex, framework-heavy applications, making manual auditing slow and costly. Large Language Models (LLMs) could help address this gap, but their auditing potential remains largely unexplored due to three limitations: (1) scarce fine-grained annotations for fair assessment; (2) abundant benign code obscuring malicious signals; and (3) unverifiable, hallucination-prone outputs undermining attribution credibility. To close this gap, we introduce MalEval, a comprehensive framework for fine-grained Android malware auditing, designed to evaluate how effectively LLMs support auditing under real-world constraints. MalEval provides expert-verified reports and an updated sensitive API list to mitigate ground truth scarcity and reduce noise via static reachability analysis. Function-level structural representations serve as intermediate attribution units for verifiable evaluation. Building on this, we define four analyst-aligned tasks -- function prioritization, evidence attribution, behavior synthesis, and sample discrimination -- together with domain-specific metrics and a unified workload-oriented score. We evaluate seven widely used LLMs on a curated dataset of recent malware and misclassified benign apps, offering the first systematic assessment of their auditing capabilities. MalEval reveals both promising potential and critical limitations across audit stages, providing a reproducible benchmark and foundation for future research on LLM-enhanced malware behavior auditing. MalEval is publicly available at https://github.com/ZhengXR930/MalEval.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14297v1">A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Safety alignment aims to prevent Large Language Models (LLMs) from responding to harmful queries. To strengthen safety protections, jailbreak methods are developed to simulate malicious attacks and uncover vulnerabilities. In this paper, we introduce HILL (Hiding Intention by Learning from LLMs), a novel jailbreak approach that systematically transforms imperative harmful requests into learning-style questions with only straightforward hypotheticality indicators. Further, we introduce two new metrics to thoroughly evaluate the utility of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong effectiveness, generalizability, and harmfulness. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. Results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. Moreover, the assessment on our constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of balancing helpfulness and safety alignments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15234v1">Exploring the Capabilities of LLM Encoders for Image-Text Retrieval in Chest X-rays</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 24 pages, 2 figures, under review
    </div>
    <details class="paper-abstract">
      Vision-language pretraining has advanced image-text alignment, yet progress in radiology remains constrained by the heterogeneity of clinical reports, including abbreviations, impression-only notes, and stylistic variability. Unlike general-domain settings where more data often leads to better performance, naively scaling to large collections of noisy reports can plateau or even degrade model learning. We ask whether large language model (LLM) encoders can provide robust clinical representations that transfer across diverse styles and better guide image-text alignment. We introduce LLM2VEC4CXR, a domain-adapted LLM encoder for chest X-ray reports, and LLM2CLIP4CXR, a dual-tower framework that couples this encoder with a vision backbone. LLM2VEC4CXR improves clinical text understanding over BERT-based baselines, handles abbreviations and style variation, and achieves strong clinical alignment on report-level metrics. LLM2CLIP4CXR leverages these embeddings to boost retrieval accuracy and clinically oriented scores, with stronger cross-dataset generalization than prior medical CLIP variants. Trained on 1.6M CXR studies from public and private sources with heterogeneous and noisy reports, our models demonstrate that robustness -- not scale alone -- is the key to effective multimodal learning. We release models to support further research in medical image-text representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14233v1">Apertus: Democratizing Open and Compliant LLMs for Global Language Environments</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting robots.txt exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23804v2">Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) achieve strong performance on text-to-SQL parsing, they sometimes exhibit unexpected failures in which they are confidently incorrect. Building trustworthy text-to-SQL systems thus requires eliciting reliable uncertainty measures from the LLM. In this paper, we study the problem of providing a calibrated confidence score that conveys the likelihood of an output query being correct. Our work is the first to establish a benchmark for post-hoc calibration of LLM-based text-to-SQL parsing. In particular, we show that Platt scaling, a canonical method for calibration, provides substantial improvements over directly using raw model output probabilities as confidence scores. Furthermore, we propose a method for text-to-SQL calibration that leverages the structured nature of SQL queries to provide more granular signals of correctness, named "sub-clause frequency" (SCF) scores. Using multivariate Platt scaling (MPS), our extension of the canonical Platt scaling technique, we combine individual SCF scores into an overall accurate and calibrated score. Empirical evaluation on two popular text-to-SQL datasets shows that our approach of combining MPS and SCF yields further improvements in calibration and the related task of error detection over traditional Platt scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14180v1">Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 24 pages, 11 figures. The paper presents a novel framework for generating a personal finance dataset. The resulting fine-tuned model and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20781v2">Using LLMs in Generating Design Rationale for Software Architecture Decisions</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 38 pages, 5 images, 9 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. To further understand the trustworthiness and applicability of LLM-generated DR in practice, we conducted semi-structured interviews with six practitioners. Based on the experimental and interview results, we discussed the pros and cons of the three prompting strategies, the strengths and limitations of LLM-generated DR, and the implications for the practical use of LLM-generated DR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14169v1">TopoSizing: An LLM-aided Framework of Topology-based Understanding and Sizing for AMS Circuits</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Analog and mixed-signal circuit design remains challenging due to the shortage of high-quality data and the difficulty of embedding domain knowledge into automated flows. Traditional black-box optimization achieves sampling efficiency but lacks circuit understanding, which often causes evaluations to be wasted in low-value regions of the design space. In contrast, learning-based methods embed structural knowledge but are case-specific and costly to retrain. Recent attempts with large language models show potential, yet they often rely on manual intervention, limiting generality and transparency. We propose TopoSizing, an end-to-end framework that performs robust circuit understanding directly from raw netlists and translates this knowledge into optimization gains. Our approach first applies graph algorithms to organize circuits into a hierarchical device-module-stage representation. LLM agents then execute an iterative hypothesis-verification-refinement loop with built-in consistency checks, producing explicit annotations. Verified insights are integrated into Bayesian optimization through LLM-guided initial sampling and stagnation-triggered trust-region updates, improving efficiency while preserving feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18325v3">Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries--a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models' safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual scenarios. We have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets are available at https://github.com/Master-PLC/RASS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08627v2">NL in the Middle: Code Translation with LLMs and Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Studies show that large language models (LLMs) produce buggy code translations. One promising avenue to improve translation accuracy is through intermediate representations, which provide structured guidance for the translation process. We investigate whether LLM-based code translation can benefit from intermediate representations, specifically in the form of natural language (NL) summaries and abstract syntax trees (ASTs). Since prompt engineering greatly affects LLM performance, we consider several ways to integrate these representations, from one-shot to chain-of-thought (CoT) prompting. Using Open GPT4 8X7B and specialized StarCoder and CodeGen models on popular code translation benchmarks (CodeNet and AVATAR), we find that CoT with an intermediate NL summary performs best, with an increase of 13.8% and 6.7%, respectively, in successful translations for the best-performing model (Open GPT4 8X7B) compared to the zero-shot prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01658v2">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 19pages, 13 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment. The code is available at https://github.com/ml-postech/CoPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18216v2">Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13978v1">LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Paper accepted in the proceedings of the ACM/IEEE Supercomputing Conference (SC). Cite it as Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, and Rafael Ferreira da Silva. 2025. LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology. In SC Workshops (WORKS)
    </div>
    <details class="paper-abstract">
      Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v3">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09974v2">Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been used in many application domains, including cyber security. The application of LLMs in the cyber security domain presents significant opportunities, such as for enhancing threat analysis and malware detection, but it can also introduce critical risks and safety concerns, including potential personal data leakage and automated generation of new malware. Building on recent findings that fine-tuning LLMs with pseudo-malicious cyber security data significantly compromises their safety, this paper presents a comprehensive validation and extension of these safety risks using a different evaluation framework. We employ the garak red teaming framework with the OWASP Top 10 for LLM Applications to assess four open-source LLMs: Mistral 7B, Llama 3 8B, Gemma 2 9B, and DeepSeek R1 8B. Our evaluation confirms and extends previous findings, showing that fine-tuning reduces safety resilience across all tested LLMs (e.g., the failure rate of Mistral 7B against prompt injection increases from 9.1% to 68.7%). We further propose and evaluate a novel safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This work validates previous safety concerns through independent evaluation and introduces new methods for mitigating these risks, contributing towards the development of secure, trustworthy, and ethically aligned LLMs. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path towards developing safer fine-tuning methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09334v3">CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13942v1">Evaluating Classical Software Process Models as Coordination Mechanisms for LLM-Based Software Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      [Background] Large Language Model (LLM)-based multi-agent systems (MAS) are transforming software development by enabling autonomous collaboration. Classical software processes such asWaterfall, V-Model, and Agile offer structured coordination patterns that can be repurposed to guide these agent interactions. [Aims] This study explores how traditional software development processes can be adapted as coordination scaffolds for LLM based MAS and examines their impact on code quality, cost, and productivity. [Method] We executed 11 diverse software projects under three process models and four GPT variants, totaling 132 runs. Each output was evaluated using standardized metrics for size (files, LOC), cost (execution time, token usage), and quality (code smells, AI- and human detected bugs). [Results] Both process model and LLM choice significantly affected system performance. Waterfall was most efficient, V-Model produced the most verbose code, and Agile achieved the highest code quality, albeit at higher computational cost. [Conclusions] Classical software processes can be effectively instantiated in LLM-based MAS, but each entails trade-offs across quality, cost, and adaptability. Process selection should reflect project goals, whether prioritizing efficiency, robustness, or structured validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01129v5">Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Automatic open-domain dialogue evaluation has attracted increasing attention, yet remains challenging due to the complexity of assessing response appropriateness. Traditional evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with contexts. However, adversarial negative responses, despite possessing high lexical overlap with contexts, can be semantically incongruous. Consequently, existing metrics struggle to effectively evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in handling adversarial negative examples. We propose a novel evaluation framework that integrates Abstract Meaning Representation (AMR) enhanced domain-specific language models (SLMs) with LLMs. Our SLMs explicitly incorporate AMR graph information through a gating mechanism for enhanced semantic representation learning, while both SLM predictions and AMR knowledge are integrated into LLM prompts for robust evaluation. Extensive experiments on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to state-of-the-art baselines. Our comprehensive ablation studies reveal that AMR graph information contributes substantially more to performance improvements. Our framework achieves strong correlations with human judgments across multiple datasets, establishing a new benchmark for dialogue evaluation. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v7">Crafting Customisable Characters with LLMs: A Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12810v1">H$^2$R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown strong potential in multi-task scenarios, owing to their ability to transfer knowledge across diverse tasks. However, existing approaches often treat prior experiences and knowledge as monolithic units, leading to inefficient and coarse-grained knowledge transfer. In this work, we propose a novel hierarchical memory architecture that enables fine-grained knowledge transfer by decoupling high-level planning memory from low-level execution memory. To construct and refine these hierarchical memories, we introduce Hierarchical Hindsight Reflection (H$^2$R), a mechanism that distills reusable and hierarchical knowledge from past agent-environment interactions. At test time, H$^2$R performs retrievals of high-level and low-level memories separately, allowing LLM-based agents to efficiently access and utilize task-relevant knowledge for new tasks.Experimental results across two benchmarks demonstrate that H$^2$R can improve generalization and decision-making performance, outperforming prior baselines such as Expel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05179v3">Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12798v1">LLM-Based Approach for Enhancing Maintainability of Automotive Architectures</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      There are many bottlenecks that decrease the flexibility of automotive systems, making their long-term maintenance, as well as updates and extensions in later lifecycle phases increasingly difficult, mainly due to long re-engineering, standardization, and compliance procedures, as well as heterogeneity and numerosity of devices and underlying software components involved. In this paper, we explore the potential of Large Language Models (LLMs) when it comes to the automation of tasks and processes that aim to increase the flexibility of automotive systems. Three case studies towards achieving this goal are considered as outcomes of early-stage research: 1) updates, hardware abstraction, and compliance, 2) interface compatibility checking, and 3) architecture modification suggestions. For proof-of-concept implementation, we rely on OpenAI's GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v3">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v3">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12750v1">What Makes a Good Generated Image? Investigating Human and Multimodal LLM Image Preference Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 7 pages, 9 figures, 3 tables; appendix 16 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automated evaluation of generative text-to-image models remains a challenging problem. Recent works have proposed using multimodal LLMs to judge the quality of images, but these works offer little insight into how multimodal LLMs make use of concepts relevant to humans, such as image style or composition, to generate their overall assessment. In this work, we study what attributes of an image--specifically aesthetics, lack of artifacts, anatomical accuracy, compositional correctness, object adherence, and style--are important for both LLMs and humans to make judgments on image quality. We first curate a dataset of human preferences using synthetically generated image pairs. We use inter-task correlation between each pair of image quality attributes to understand which attributes are related in making human judgments. Repeating the same analysis with LLMs, we find that the relationships between image quality attributes are much weaker. Finally, we study individual image quality attributes by generating synthetic datasets with a high degree of control for each axis. Humans are able to easily judge the quality of an image with respect to all of the specific image quality attributes (e.g. high vs. low aesthetic image), however we find that some attributes, such as anatomical accuracy, are much more difficult for multimodal LLMs to learn to judge. Taken together, these findings reveal interesting differences between how humans and multimodal LLMs perceive images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12743v1">Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      We propose a new, training-free method, Graph Reasoning via Retrieval Augmented Framework (GRRAF), that harnesses retrieval-augmented generation (RAG) alongside the code-generation capabilities of large language models (LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target graph is stored in a graph database, and the LLM is prompted to generate executable code queries that retrieve the necessary information. This approach circumvents the limitations of existing methods that require extensive finetuning or depend on predefined algorithms, and it incorporates an error feedback loop with a time-out mechanism to ensure both correctness and efficiency. Experimental evaluations on the GraphInstruct dataset reveal that GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle detection, bipartite graph checks, shortest path computation, and maximum flow, while maintaining consistent token costs regardless of graph sizes. Imperfect but still very high performance is observed on subgraph matching. Notably, GRRAF scales effectively to large graphs with up to 10,000 nodes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19345v2">PatentScore: Multi-dimensional Evaluation of LLM-Generated Patent Claims</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      High-stakes texts such as patent claims, medical records, and technical reports are structurally complex and demand a high degree of reliability and precision. While large language models (LLMs) have recently been applied to automate their generation in high-stakes domains, reliably evaluating such outputs remains a major challenge. Conventional natural language generation (NLG) metrics are effective for generic documents but fail to capture the structural and legal characteristics essential to evaluating complex high-stakes documents. To address this gap, we propose PatentScore, a multi-dimensional evaluation framework specifically designed for one of the most intricate and rigorous domains, patent claims. PatentScore integrates hierarchical decomposition of claim elements, validation patterns grounded in legal and technical standards, and scoring across structural, semantic, and legal dimensions. In experiments on our dataset which consists of 400 Claim1, PatentScore achieved the highest correlation with expert annotations ($r = 0.819$), significantly outperforming widely used NLG metrics. This work establishes a new standard for evaluating LLM-generated patent claims, providing a solid foundation for research on patent generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10259v2">Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of large language model (LLM) services imposes increasing demands on distributed GPU inference infrastructure. Most existing scheduling systems follow a reactive paradigm, relying solely on the current system state to make decisions, without considering how task demand and resource availability evolve over time. This lack of temporal awareness in reactive approaches leads to inefficient GPU utilization, high task migration overhead, and poor system responsiveness under dynamic workloads. In this work, we identify the fundamental limitations of these instantaneous-state-only scheduling approaches and propose Temporal Optimal Resource scheduling via Two-layer Architecture (TORTA). TORTA introduces a spatiotemporal scheduling framework that captures both long-term workload patterns and short-term execution constraints. It adopts a two-layer design: a macro-level scheduler leverages reinforcement learning and optimal transport to coordinate inter-region task distribution, while a micro-level allocator refines task-to-server assignments within each region to reduce latency and switching costs. Experimental results across multiple network topologies show that TORTA reduces average inference response time by up to 15\%, improves load balance by approximately 4-5\%, and cuts total operational cost by 10-20\% compared to state-of-the-art baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18530v3">IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12678v1">Instance-level Randomization: Toward More Stable LLM Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Evaluations of large language models (LLMs) suffer from instability, where small changes of random factors such as few-shot examples can lead to drastic fluctuations of scores and even model rankings. Moreover, different LLMs can have different preferences for a certain setting of random factors. As a result, using a fixed setting of random factors, which is often adopted as the paradigm of current evaluations, can lead to potential unfair comparisons between LLMs. To mitigate the volatility of evaluations, we first theoretically analyze the sources of variance induced by changes in random factors. Targeting these specific sources, we then propose the instance-level randomization (ILR) method to reduce variance and enhance fairness in model comparisons. Instead of using a fixed setting across the whole benchmark in a single experiment, we randomize all factors that affect evaluation scores for every single instance, run multiple experiments and report the averaged score. Theoretical analyses and empirical results demonstrate that ILR can reduce the variance and unfair comparisons caused by random factors, as well as achieve similar robustness level with less than half computational cost compared with previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12672v1">Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12649v1">A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12638v1">FinSentLLM: Multi-LLM and Structured Semantic Signals for Enhanced Financial Sentiment Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis (FSA) has attracted significant attention, and recent studies increasingly explore large language models (LLMs) for this field. Yet most work evaluates only classification metrics, leaving unclear whether sentiment signals align with market behavior. We propose FinSentLLM, a lightweight multi-LLM framework that integrates an expert panel of sentiment forecasting LLMs, and structured semantic financial signals via a compact meta-classifier. This design captures expert complementarity, semantic reasoning signal, and agreement/divergence patterns without costly retraining, yielding consistent 3-6% gains over strong baselines in accuracy and F1-score on the Financial PhraseBank dataset. In addition, we also provide econometric evidence that financial sentiment and stock markets exhibit statistically significant long-run comovement, applying Dynamic Conditional Correlation GARCH (DCC-GARCH) and the Johansen cointegration test to daily sentiment scores computed from the FNSPID dataset and major stock indices. Together, these results demonstrate that FinSentLLM delivers superior forecasting accuracy for financial sentiment and further establish that sentiment signals are robustly linked to long-run equity market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22388v3">Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP 2025 Main, Oral
    </div>
    <details class="paper-abstract">
      LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future. DSDBench is publicly available at github.com/KevinCL16/DSDBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12625v1">ECG-aBcDe: Overcoming Model Dependence, Encoding ECG into a Universal Language for Any LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 14pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold significant promise for electrocardiogram (ECG) analysis, yet challenges remain regarding transferability, time-scale information learning, and interpretability. Current methods suffer from model-specific ECG encoders, hindering transfer across LLMs. Furthermore, LLMs struggle to capture crucial time-scale information inherent in ECGs due to Transformer limitations. And their black-box nature limits clinical adoption. To address these limitations, we introduce ECG-aBcDe, a novel ECG encoding method that transforms ECG signals into a universal ECG language readily interpretable by any LLM. By constructing a hybrid dataset of ECG language and natural language, ECG-aBcDe enables direct fine-tuning of pre-trained LLMs without architectural modifications, achieving "construct once, use anywhere" capability. Moreover, the bidirectional convertibility between ECG and ECG language of ECG-aBcDe allows for extracting attention heatmaps from ECG signals, significantly enhancing interpretability. Finally, ECG-aBcDe explicitly represents time-scale information, mitigating Transformer limitations. This work presents a new paradigm for integrating ECG analysis with LLMs. Compared with existing methods, our method achieves competitive performance on ROUGE-L and METEOR. Notably, it delivers significant improvements in the BLEU-4, with improvements of 2.8 times and 3.9 times in in-dataset and cross-dataset evaluations, respectively, reaching scores of 42.58 and 30.76. These results provide strong evidence for the feasibility of the new paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08426v6">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to IEEE TKDE
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12610v1">ScaleDoc: Scaling LLM-based Predicates over Large Document Collections</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Predicates are foundational components in data analysis systems. However, modern workloads increasingly involve unstructured documents, which demands semantic understanding, beyond traditional value-based predicates. Given enormous documents and ad-hoc queries, while Large Language Models (LLMs) demonstrate powerful zero-shot capabilities, their high inference cost leads to unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel system that addresses this by decoupling predicate execution into an offline representation phase and an optimized online filtering phase. In the offline phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations for each document. Online, for each query, it trains a lightweight proxy model on these representations to filter the majority of documents, forwarding only the ambiguous cases to the LLM for final decision. Furthermore, \textsc{ScaleDoc} proposes two core innovations to achieve significant efficiency: (1) a contrastive-learning-based framework that trains the proxy model to generate reliable predicating decision scores; (2) an adaptive cascade mechanism that determines the effective filtering policy while meeting specific accuracy targets. Our evaluations across three datasets demonstrate that \textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces expensive LLM invocations by up to 85\%, making large-scale semantic analysis practical and efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11177v2">Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11967v2">MillStone: How Open-Minded Are LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 19 pages, 7 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12590v1">DPCheatSheet: Using Worked and Erroneous LLM-usage Examples to Scaffold Differential Privacy Implementation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      This paper explores how programmers without specialized expertise in differential privacy (DP) (i.e., novices) can leverage LLMs to implement DP programs with minimal training. We first conducted a need-finding study with 6 novices and 3 experts to understand how they utilize LLMs in DP implementation. While DP experts can implement correct DP analyses through a few prompts, novices struggle to articulate their requirements in prompts and lack the skills to verify the correctness of the generated code. We then developed DPCheatSheet, an instructional tool that helps novices implement DP using LLMs. DPCheatSheet combines two learning concepts: it annotates an expert's workflow with LLMs as a worked example to bridge the expert mindset to novices, and it presents five common mistakes in LLM-based DP code generation as erroneous examples to support error-driven learning. We demonstrated the effectiveness of DPCheatSheet with an error identification study and an open-ended DP implementation study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17539v3">Breaking Single-Tester Limits: Multi-Agent LLMs for Multi-User Feature Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to International Conference on Software Engineering (ICSE 2026). arXiv admin note: substantial text overlap with arXiv:2504.15474
    </div>
    <details class="paper-abstract">
      The growing dependence on mobile phones and their apps has made multi-user interactive features, like chat calls, live streaming, and video conferencing, indispensable for bridging the gaps in social connectivity caused by physical and situational barriers. However, automating these interactive features for testing is fraught with challenges, owing to their inherent need for timely, dynamic, and collaborative user interactions, which current automated testing methods inadequately address. Inspired by the concept of agents designed to autonomously and collaboratively tackle problems, we propose MAdroid, a novel multi-agent approach powered by the Large Language Models (LLMs) to automate the multi-user interactive task for app feature testing. Specifically, MAdroid employs two functional types of multi-agents: user agents (Operator) and supervisor agents (Coordinator and Observer). Each agent takes a specific role: the Coordinator directs the interactive task; the Operator mimics user interactions on the device; and the Observer monitors and reviews the task automation process. Our evaluation, which included 41 multi-user interactive tasks, demonstrates the effectiveness of our approach, achieving 82.9% of the tasks with 96.8% action similarity, outperforming the ablation studies and state-of-the-art baselines. Additionally, a preliminary investigation underscores MAdroid's practicality by helping identify 11 multi-user interactive bugs during regression app testing, confirming its potential value in real-world software development contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12546v1">Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Face forgery detection faces a critical challenge: a persistent gap between offline benchmarks and real-world efficacy,which we attribute to the ecological invalidity of training data.This work introduces Agent4FaceForgery to address two fundamental problems: (1) how to capture the diverse intents and iterative processes of human forgery creation, and (2) how to model the complex, often adversarial, text-image interactions that accompany forgeries in social media. To solve this,we propose a multi-agent framework where LLM-poweredagents, equipped with profile and memory modules, simulate the forgery creation process. Crucially, these agents interact in a simulated social environment to generate samples labeled for nuanced text-image consistency, moving beyond simple binary classification. An Adaptive Rejection Sampling (ARS) mechanism ensures data quality and diversity. Extensive experiments validate that the data generated by our simulationdriven approach brings significant performance gains to detectors of multiple architectures, fully demonstrating the effectiveness and value of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08256v2">FIER: Fine-Grained and Efficient KV Cache Retrieval for Long-context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP2025 Camera-ready
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache reading latency increases significantly with context lengths, hindering the efficiency of long-context LLM inference. To address this, previous works propose retaining a small fraction of KV cache based on token importance. For example, KV eviction uses static heuristics to retain tokens, while KV retrieval dynamically selects query-relevant tokens for more adaptive cache management. However, we observe that important tokens are often sparsely distributed across the long context. This sparsity makes existing page-level KV retrieval inaccurate, as each page may include irrelevant tokens and miss critical ones. In this work, we propose Fier, a \underline{Fi}ne-Grained and \underline{E}fficient KV cache \underline{R}etrieval method. Fier uses 1-bit quantized keys to estimate the importance of each token, resulting in efficient and precise retrieval. Experiments show that Fier matches full KV performance using only 11\% of the cache budget across various long-context tasks, reducing decoding latency by 1.2$\times$ to 1.5$\times$.Code is available at https://github.com/SimWangArizona/FIER
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13557v1">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process. In this work, we propose MACO -- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process. We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12274v2">Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Real-time multi-robot coordination in hazardous and adversarial environments requires fast, reliable adaptation to dynamic threats. While Large Language Models (LLMs) offer strong high-level reasoning capabilities, the lack of safety guarantees limits their direct use in critical decision-making. In this paper, we propose a hierarchical optimization framework that integrates LLMs into the decision loop for multi-robot target tracking in dynamic and hazardous environments. Rather than generating control actions directly, LLMs are used to generate task configuration and adjust parameters in a bi-level task allocation and planning problem. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, demonstrating its effectiveness and showcasing its capability for safe LLM integration for multi-robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13514v1">AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by the 24th IEEE International Conference on Machine Learning and Applications (ICMLA'25)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13487v1">Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Developing reliable data enrichment pipelines demands significant engineering expertise. We present Prompt2DAG, a methodology that transforms natural language descriptions into executable Apache Airflow DAGs. We evaluate four generation approaches -- Direct, LLM-only, Hybrid, and Template-based -- across 260 experiments using thirteen LLMs and five case studies to identify optimal strategies for production-grade automation. Performance is measured using a penalized scoring framework that combines reliability with code quality (SAT), structural integrity (DST), and executability (PCT). The Hybrid approach emerges as the optimal generative method, achieving a 78.5% success rate with robust quality scores (SAT: 6.79, DST: 7.67, PCT: 7.76). This significantly outperforms the LLM-only (66.2% success) and Direct (29.2% success) methods. Our findings show that reliability, not intrinsic code quality, is the primary differentiator. Cost-effectiveness analysis reveals the Hybrid method is over twice as efficient as Direct prompting per successful DAG. We conclude that a structured, hybrid approach is essential for balancing flexibility and reliability in automated workflow generation, offering a viable path to democratize data pipeline development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13471v1">An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 To appear at ICSE 26. 12 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v1">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19749v2">What's Not Said Still Hurts: A Description-Based Evaluation Framework for Measuring Social Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit social biases inherited from their training data. While existing benchmarks evaluate bias by term-based mode through direct term associations between demographic terms and bias terms, LLMs have become increasingly adept at avoiding biased responses, leading to seemingly low levels of bias. However, biases persist in subtler, contextually hidden forms that traditional benchmarks fail to capture. We introduce the Description-based Bias Benchmark (DBB), a novel dataset designed to assess bias at the semantic level that bias concepts are hidden within naturalistic, subtly framed contexts in real-world scenarios rather than superficial terms. We analyze six state-of-the-art LLMs, revealing that while models reduce bias in response at the term level, they continue to reinforce biases in nuanced settings. Data, code, and results are available at https://github.com/JP-25/Description-based-Bias-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14289v1">From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14285v1">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14275v1">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 (e.g.: 18 pages, 6 figures, 6 tables)
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14274v1">Discovering New Theorems via LLMs with In-Context Proof Learning in Lean</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated significant promise in formal theorem proving. However, previous works mainly focus on solving existing problems. In this paper, we focus on the ability of LLMs to find novel theorems. We propose Conjecturing-Proving Loop pipeline for automatically generating mathematical conjectures and proving them in Lean 4 format. A feature of our approach is that we generate and prove further conjectures with context including previously generated theorems and their proofs, which enables the generation of more difficult proofs by in-context learning of proof strategies without changing parameters of LLMs. We demonstrated that our framework rediscovered theorems with verification, which were published in past mathematical papers and have not yet formalized. Moreover, at least one of these theorems could not be proved by the LLM without in-context learning, even in natural language, which means that in-context learning was effective for neural theorem proving. The source code is available at https://github.com/auto-res/ConjecturingProvingLoop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14273v1">Automated and Context-Aware Code Documentation Leveraging Advanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Code documentation is essential to improve software maintainability and comprehension. The tedious nature of manual code documentation has led to much research on automated documentation generation. Existing automated approaches primarily focused on code summarization, leaving a gap in template-based documentation generation (e.g., Javadoc), particularly with publicly available Large Language Models (LLMs). Furthermore, progress in this area has been hindered by the lack of a Javadoc-specific dataset that incorporates modern language features, provides broad framework/library coverage, and includes necessary contextual information. This study aims to address these gaps by developing a tailored dataset and assessing the capabilities of publicly available LLMs for context-aware, template-based Javadoc generation. In this work, we present a novel, context-aware dataset for Javadoc generation that includes critical structural and semantic information from modern Java codebases. We evaluate five open-source LLMs (including LLaMA-3.1, Gemma-2, Phi-3, Mistral, Qwen-2.5) using zero-shot, few-shot, and fine-tuned setups and provide a comparative analysis of their performance. Our results demonstrate that LLaMA 3.1 performs consistently well and is a reliable candidate for practical, automated Javadoc generation, offering a viable alternative to proprietary systems.
    </details>
</div>
