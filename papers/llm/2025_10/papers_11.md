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
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12389v1">Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 6 pages 4 figures
    </div>
    <details class="paper-abstract">
      Tokenization disparities pose a significant barrier to achieving equitable access to artificial intelligence across linguistically diverse populations. This study conducts a large-scale cross-linguistic evaluation of tokenization efficiency in over 200 languages to systematically quantify computational inequities in large language models (LLMs). Using a standardized experimental framework, we applied consistent preprocessing and normalization protocols, followed by uniform tokenization through the tiktoken library across all language samples. Comprehensive tokenization statistics were collected using established evaluation metrics, including Tokens Per Sentence (TPS) and Relative Tokenization Cost (RTC), benchmarked against English baselines. Our cross-linguistic analysis reveals substantial and systematic disparities: Latin-script languages consistently exhibit higher tokenization efficiency, while non-Latin and morphologically complex languages incur significantly greater token inflation, often 3-5 times higher RTC ratios. These inefficiencies translate into increased computational costs and reduced effective context utilization for underrepresented languages. Overall, the findings highlight structural inequities in current AI systems, where speakers of low-resource and non-Latin languages face disproportionate computational disadvantages. Future research should prioritize the development of linguistically informed tokenization strategies and adaptive vocabulary construction methods that incorporate typological diversity, ensuring more inclusive and computationally equitable multilingual AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12367v1">LLM-REVal: Can We Trust LLM Reviewers Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation. This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments: (1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones; (2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions. Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements. These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12355v1">Fine-grained Analysis of Brain-LLM Alignment through Input Attribution</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Understanding the alignment between large language models (LLMs) and human brain activity can reveal computational principles underlying language processing. We introduce a fine-grained input attribution method to identify the specific words most important for brain-LLM alignment, and leverage it to study a contentious research question about brain-LLM alignment: the relationship between brain alignment (BA) and next-word prediction (NWP). Our findings reveal that BA and NWP rely on largely distinct word subsets: NWP exhibits recency and primacy biases with a focus on syntax, while BA prioritizes semantic and discourse-level information with a more targeted recency effect. This work advances our understanding of how LLMs relate to human language processing and highlights differences in feature reliance between BA and NWP. Beyond this study, our attribution method can be broadly applied to explore the cognitive relevance of model predictions in diverse language processing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12350v1">O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12306v1">A large-scale, unsupervised pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable, unsupervised pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English consider construction. Using GPT-5 through the OpenAI API, we annotate 143,933 sentences from the Corpus of Historical American English (COHA) in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, opening new possibilities for corpus-based research, though implementation requires attention to costs, licensing, and other ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12294v1">Show Your Title! A Scoping Review on Verbalization in Software Engineering with LLM-Assisted Screening</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 preprint of a paper under publication in Quality of Information and Communications Technology 2025
    </div>
    <details class="paper-abstract">
      Understanding how software developers think, make decisions, and behave remains a key challenge in software engineering (SE). Verbalization techniques (methods that capture spoken or written thought processes) offer a lightweight and accessible way to study these cognitive aspects. This paper presents a scoping review of research at the intersection of SE and psychology (PSY), focusing on the use of verbal data. To make large-scale interdisciplinary reviews feasible, we employed a large language model (LLM)-assisted screening pipeline using GPT to assess the relevance of over 9,000 papers based solely on titles. We addressed two questions: what themes emerge from verbalization-related work in SE, and how effective are LLMs in supporting interdisciplinary review processes? We validated GPT's outputs against human reviewers and found high consistency, with a 13\% disagreement rate. Prominent themes mainly were tied to the craft of SE, while more human-centered topics were underrepresented. The data also suggests that SE frequently draws on PSY methods, whereas the reverse is rare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26072v2">The Silent Judge: Unacknowledged Shortcut Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 9 Pages, 5 Tables, 1 Figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as automatic judges to evaluate system outputs in tasks such as summarization, dialogue, and creative writing. A faithful judge should base its verdicts solely on response quality and explicitly acknowledge the factors shaping its decision. We show that current LLM judges fail on both counts by relying on shortcuts introduced in the prompt. Our study uses two evaluation datasets: ELI5, a benchmark for long-form question answering, and LitBench, a recent benchmark for creative writing. Both datasets provide pairwise comparisons, where the evaluator must choose which of two responses is better. From each dataset we construct 100 pairwise judgment tasks and employ two widely used models, GPT-4o and Gemini-2.5-Flash, as evaluators in the role of LLM-as-a-judge. For each pair, we assign superficial cues to the responses, provenance cues indicating source identity (Human, Expert, LLM, or Unknown) and recency cues indicating temporal origin (Old, 1950 vs. New, 2025), while keeping the rest of the prompt fixed. Results reveal consistent verdict shifts: both models exhibit a strong recency bias, systematically favoring new responses over old, as well as a clear provenance hierarchy (Expert > Human > LLM > Unknown). These biases are especially pronounced in GPT-4o and in the more subjective and open-ended LitBench domain. Crucially, cue acknowledgment is rare: justifications almost never reference the injected cues, instead rationalizing decisions in terms of content qualities. These findings demonstrate that current LLM-as-a-judge systems are shortcut-prone and unfaithful, undermining their reliability as evaluators in both research and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26041v2">Unspoken Hints: Accuracy Without Acknowledgement in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 5 Pages, 4 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on chain-of-thought (CoT) prompting to solve mathematical and logical reasoning tasks. Yet, a central question remains: to what extent are these generated rationales \emph{faithful} to the underlying computations, rather than post-hoc narratives shaped by hints that function as answer shortcuts embedded in the prompt? Following prior work on hinted vs.\ unhinted prompting, we present a systematic study of CoT faithfulness under controlled hint manipulations. Our experimental design spans four datasets (AIME, GSM-Hard, MATH-500, UniADILR), two state-of-the-art models (GPT-4o and Gemini-2-Flash), and a structured set of hint conditions varying in correctness (correct and incorrect), presentation style (sycophancy and data leak), and complexity (raw answers, two-operator expressions, four-operator expressions). We evaluate both task accuracy and whether hints are explicitly acknowledged in the reasoning. Our results reveal three key findings. First, correct hints substantially improve accuracy, especially on harder benchmarks and logical reasoning, while incorrect hints sharply reduce accuracy in tasks with lower baseline competence. Second, acknowledgement of hints is highly uneven: equation-based hints are frequently referenced, whereas raw hints are often adopted silently, indicating that more complex hints push models toward verbalizing their reliance in the reasoning process. Third, presentation style matters: sycophancy prompts encourage overt acknowledgement, while leak-style prompts increase accuracy but promote hidden reliance. This may reflect RLHF-related effects, as sycophancy exploits the human-pleasing side and data leak triggers the self-censoring side. Together, these results demonstrate that LLM reasoning is systematically shaped by shortcuts in ways that obscure faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12255v1">Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Dataset and code: https://huggingface.co/datasets/dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust ; https://github.com/bmanczak/MedQA-MultiTurnRobustness Accepted as a poster at NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly transitioning into medical clinical use, yet their reliability under realistic, multi-turn interactions remains poorly understood. Existing evaluation frameworks typically assess single-turn question answering under idealized conditions, overlooking the complexities of medical consultations where conflicting input, misleading context, and authority influence are common. We introduce MedQA-Followup, a framework for systematically evaluating multi-turn robustness in medical question answering. Our approach distinguishes between shallow robustness (resisting misleading initial context) and deep robustness (maintaining accuracy when answers are challenged across turns), while also introducing an indirect-direct axis that separates contextual framing (indirect) from explicit suggestion (direct). Using controlled interventions on the MedQA dataset, we evaluate five state-of-the-art LLMs and find that while models perform reasonably well under shallow perturbations, they exhibit severe vulnerabilities in multi-turn settings, with accuracy dropping from 91.2% to as low as 13.5% for Claude Sonnet 4. Counterintuitively, indirect, context-based interventions are often more harmful than direct suggestions, yielding larger accuracy drops across models and exposing a significant vulnerability for clinical deployment. Further compounding analyses reveal model differences, with some showing additional performance drops under repeated interventions while others partially recovering or even improving. These findings highlight multi-turn robustness as a critical but underexplored dimension for safe and reliable deployment of medical LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v3">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12245v1">MoRA: On-the-fly Molecule-aware Low-Rank Adaptation Framework for LLM-based Multi-Modal Molecular Assistant</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Effectively integrating molecular graph structures with Large Language Models (LLMs) is a key challenge in drug discovery. Most existing multi-modal alignment methods typically process these structures by fine-tuning the LLM or adding a static adapter simultaneously. However, these approaches have two main limitations: (1) it optimizes a shared parameter space across all molecular inputs, limiting the model's ability to capture instance-specific structural features; and (2) fine-tuning the LLM for molecular tasks can lead to catastrophic forgetting, undermining its general reasoning capabilities. In this paper, instead of static task-oriented adaptation, we propose an instance-specific parameter space alignment approach for each molecule on-the-fly. To this end, we introduce Molecule-aware Low-Rank Adaptation (MoRA) that produces a unique set of low-rank adaptation weights for each input molecular graph. These weights are then dynamically injected into a frozen LLM, allowing the model to adapt its reasoning to the structure of each molecular input, while preserving the LLM's core knowledge. Extensive experiments demonstrate that on key molecular tasks, such as chemical reaction prediction and molecular captioning, MoRA's instance-specific dynamic adaptation outperforms statically adapted baselines, including a 14.1% relative improvement in reaction prediction exact match and a 22% reduction in error for quantum property prediction. The code is available at https://github.com/jk-sounds/MoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12229v1">Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11104v1">Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Current approaches for strengthening LLM reasoning tend to introduce a training bias toward human-like reasoning trajectories. In step-wise preference optimization, in particular, dependence on human or higher-capacity model annotations for intermediate steps limits exploration of alternative, non-human-like reasoning paths and thus constrains achievable performance. Furthermore, through a small-scale pilot study, we observed that in approximately 75% of cases, the model's first erroneous step occurs after the lowest-confidence point. This suggests that guiding the model at its lowest-confidence point before an error provides more accurate supervision than locating the first explicit error. In this paper, we propose Confidence-Guided Reasoning Path Preference Optimization (CGPO), a method that leverages a confidence signal to identify points of maximal uncertainty in the model's reasoning process and applies self-generated, non-human-like reasoning-path guidance to mitigate trajectory drift. Our experiments span diverse models applied to both code and mathematical reasoning tasks. The results show that, with the same amount of training data, our method using data generated by a small model can achieve better performance in most cases compared with approaches using data generated by a strong model or human-annotated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11076v1">DebugTA: An LLM-Based Agent for Simplifying Debugging and Teaching in Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      In programming education, Debugging and Teaching (DT) task is a common scenario where students receive assistance in correcting their erroneous code. The task involves multiple inputs, including erroneous code, error messages, reference solutions, and the question description, with the goal of generating modification suggestions to the erroneous code. However, two key challenges hinder the effectiveness of existing approaches. Firstly, the complexity and heterogeneity of inputs inherent in DT tasks significantly elevate the reasoning challenges faced by LLMs. Second, existing approaches often fail to fully leverage the availability of standard code in DT tasks, forcing models to rely solely on complex multi-step reasoning, which limits the potential of LLMs in addressing DT tasks effectively. To address these challenges, we propose DebugTA, a novel LLM-based debugging and teaching agent with specialized tools for standard code retrieval, variable substitution to align reference code, and an external compiler for real-time code analysis. Guided by explicit pedagogical and debugging principles, DebugTA acts as an agent that decomposes a complex task into sequential LLM interactions, each utilizing distinct tools for specific subtasks, thereby simplifying the logical reasoning at each step and reducing overall reasoning complexity. Furthermore, DebugTA utilizes tool calls to align the standard code with the erroneous code as much as possible, allowing the LLM to focus on logic errors within the erroneous code and improving the accuracy of the generated suggestions. To rigorously assess the quality of modification suggestions, we introduce a student simulator-teacher interaction paradigm. Experimental results on three real-world code datasets demonstrate that DebugTA consistently improves teaching effectiveness while significantly reducing computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v3">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12839v1">FaStFACT: Faster, Stronger Long-Form Factuality Evaluations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to accuracy issues and costly human assessment. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to complex pipeline components unsuitable for long LLM outputs, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence collection of one-line snippets. To address these limitations, we propose \name, a fast and strong evaluation framework that achieves the highest alignment with human evaluation and efficiency among existing baselines. \name first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the cost of web searching and inference calling while ensuring reliability. For searching and verification, it collects document-level evidence from crawled webpages and selectively retrieves it during verification, addressing the evidence insufficiency problem in previous pipelines. Extensive experiments based on an aggregated and manually annotated benchmark demonstrate the reliability of \name in both efficiently and effectively evaluating the factuality of long-form LLM generations. Code and benchmark data is available at https://github.com/Yingjia-Wan/FastFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12836v1">BanglaMATH : A Bangla benchmark dataset for testing LLM mathematical reasoning at grades 6, 7, and 8</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have tremendous potential to play a key role in supporting mathematical reasoning, with growing use in education and AI research. However, most existing benchmarks are limited to English, creating a significant gap for low-resource languages. For example, Bangla is spoken by nearly 250 million people who would collectively benefit from LLMs capable of native fluency. To address this, we present BanglaMATH, a dataset of 1.7k Bangla math word problems across topics such as Arithmetic, Algebra, Geometry, and Logical Reasoning, sourced from Bangla elementary school workbooks and annotated with details like grade level and number of reasoning steps. We have designed BanglaMATH to evaluate the mathematical capabilities of both commercial and open-source LLMs in Bangla, and we find that Gemini 2.5 Flash and DeepSeek V3 are the only models to achieve strong performance, with $\ge$ 80\% accuracy across three elementary school grades. Furthermore, we assess the robustness and language bias of these top-performing LLMs by augmenting the original problems with distracting information, and translating the problems into English. We show that both LLMs fail to maintain robustness and exhibit significant performance bias in Bangla. Our study underlines current limitations of LLMs in handling arithmetic and mathematical reasoning in low-resource languages, and highlights the need for further research on multilingual and equitable mathematical understanding. Dataset link: \href{https://github.com/TabiaTanzin/BanglaMATH-A-Bangla-benchmark-dataset-for-testing-LLM-mathematical-reasoning-at-grades-6-7-and-8.git}{https://github.com/BanglaMATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12835v1">Repurposing Annotation Guidelines to Instruct LLM Annotators: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 11 pages, 2 figures, 3 tables, This is a preprint of the article accepted at NLDB 2025 (Springer LNCS). The final version is available at https://doi.org/10.1007/978-3-031-97144-0_13
    </div>
    <details class="paper-abstract">
      This study investigates how existing annotation guidelines can be repurposed to instruct large language model (LLM) annotators for text annotation tasks. Traditional guidelines are written for human annotators who internalize training, while LLMs require explicit, structured instructions. We propose a moderation-oriented guideline repurposing method that transforms guidelines into clear directives for LLMs through an LLM moderation process. Using the NCBI Disease Corpus as a case study, our experiments show that repurposed guidelines can effectively guide LLM annotators, while revealing several practical challenges. The results highlight the potential of this workflow to support scalable and cost-effective refinement of annotation guidelines and automated annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v4">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Need to address additional data or methodological concerns
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine reasoning or superficial pattern recognition. Common evaluation methods, which focus on the either the final answer or the reasoning process, fail to assess the entire problem-solving procedure. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework, together with its corresponding benchmark, SMART-Bench. SMART decomposes the entire problem solving process into four distinct cognitive dimensions: Understanding, Reasoning, Arithmetic, and Reflection \& Refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings reveal genuine weaknesses in current LLMs and motivate a new metric, the All-Pass Score, to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11062v1">Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11056v1">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11040v1">Enabling Doctor-Centric Medical AI with LLMs through Workflow-Aligned Tasks and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has transformed healthcare by offering clinical guidance, yet their direct deployment to patients poses safety risks due to limited domain expertise. To mitigate this, we propose repositioning LLMs as clinical assistants that collaborate with experienced physicians rather than interacting with patients directly. We conduct a two-stage inspiration-feedback survey to identify real-world needs in clinical workflows. Guided by this, we construct DoctorFLAN, a large-scale Chinese medical dataset comprising 92,000 Q&A instances across 22 clinical tasks and 27 specialties. To evaluate model performance in doctor-facing applications, we introduce DoctorFLAN-test (550 single-turn Q&A items) and DotaBench (74 multi-turn conversations). Experimental results with over ten popular LLMs demonstrate that DoctorFLAN notably improves the performance of open-source LLMs in medical contexts, facilitating their alignment with physician workflows and complementing existing patient-oriented models. This work contributes a valuable resource and framework for advancing doctor-centered medical LLM development
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12575v2">DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at https://github.com/whfeLingYu/DemonAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05431v2">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25279v2">RL in the Wild: Characterizing RLVR Training in LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 20 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used across many domains. With their rapid development, Reinforcement Learning with Verifiable Rewards (RLVR) has surged in recent months to enhance their reasoning and understanding abilities. However, its complex data flows and diverse tasks pose substantial challenges to RL training systems, and there is limited understanding of RLVR from a system perspective. To thoroughly understand the system challenges introduced by RLVR, we present a characterization study of RLVR tasks in our LLM deployment. Specifically, we investigate the distribution and variation trends of workloads across different RL tasks across training steps. We identify issues such as GPU idling caused by skewed sequence length distribution, inefficient parallel strategies in dynamically varying workloads, inefficient data management mechanisms, and load imbalance. We describe our observations and call for further investigation into the remaining open challenges. Furthermore, we propose PolyTrace benchmark suite to conduct evaluation with realistic workloads, and a practical use case validates that PolyTrace benchmark suite exhibits 94.7% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10998v1">ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 28 pages, 11 figures, 16 tables. In submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were further amplified by sharp increases in intersectional harms (e.g., Tokenism) for gender and caste-marginalized disabled candidates, highlighting critical blind spots in current safety tools and the need for intersectional safety evaluations of frontier models in high-stakes domains like hiring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10987v1">DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 14 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at https://github.com/hsannn/ditto.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10978v1">Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Language Bias, whereby the model over-relies on auxiliary tokens-such as task descriptions and prefix-generated tokens-while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating language bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates language bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04492v2">Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The advancement of LLMs and their accessibility have triggered renewed interest in multi-agent reinforcement learning as robust and adaptive frameworks for dynamically changing environments. This paper introduces RL-Focal, a two-stage RL agent framework that routes and ensembles LLMs. First, we develop the Decider RL-agent, which learns to dynamically select an ensemble of small size ($m_i$) among $N$ LLMs ($m_i \ll N$) for incoming queries from a user-defined downstream task $i$, by maximizing both error-diversity and reasoning-performance of the selected ensemble through iterative updates of task-adaptive rewards and policy. Second, to enable effective fusion of dynamically selected LLMs, we develop the stage-2 Fusion RL-agent, which learns to resolve reasoning conflicts from different LLMs and dynamically adapts to different ensemble teams composed by the Decider Agent for different downstream tasks. Third, we introduce the focal diversity metric to better model the error correlations among multiple LLMs, further improving the generalization performance of the Decider Agent, which actively prunes the ensemble combinations. By focal diversity, we enhance performance across tasks by effectively promoting reward-aware and policy-adaptive ensemble selection and inference fusion. Extensive evaluations on five benchmarks show that RL-Focal achieves the performance improvement of 8.48\% with an ensemble of small size compared to the best individual LLM in a pool and offers stronger robustness. Code is available at https://github.com/sftekin/rl-focal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v1">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10955v1">HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a surge of research on leveraging large language models (LLMs) for sequential recommendation. LLMs have demonstrated remarkable potential in inferring users' nuanced preferences through fine-grained semantic reasoning. However, they also exhibit a notable limitation in effectively modeling collaborative signals, i.e., behavioral correlations inherent in users' historical interactions. Our empirical analysis further reveals that the attention mechanisms in LLMs tend to disproportionately focus on tokens within the same item, thereby impeding the capture of cross-item correlations. To address this limitation, we propose a novel hierarchical attention masking strategy for LLM-based recommendation, termed HatLLM. Specifically, in shallow layers, HatLLM masks attention between tokens from different items, facilitating intra-item semantic understanding; in contrast, in deep layers, HatLLM masks attention within items, thereby compelling the model to capture cross-item correlations. This progressive, layer-wise approach enables LLMs to jointly model both token-level and item-level dependencies. Extensive experiments on three real-world datasets demonstrate that HatLLM achieves significant performance gains (9.13% on average) over existing LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09393v2">ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender systems. Existing approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity users. To address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity users. With a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce platform. ChoirRec improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v2">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD dot com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD dot com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v4">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15507v5">Steering LLMs for Formal Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent advances in automated theorem proving use Large Language Models (LLMs) to translate informal mathematical statements into formal proofs. However, informal cues are often ambiguous or lack strict logical structure, making it hard for models to interpret them precisely. While existing methods achieve strong performance, little is known about how LLMs internally represent informal cues, or how these influence proof generation. To address this, we explore \textit{activation steering}, an inference-time intervention that identifies linear directions in residual activations associated with informal reasoning traces and adjusts them to improve proof construction without fine-tuning. This mechanism also yields interpretable information about how reasoning is internally encoded in the activation space of LLMs. We test our method for generating formal proofs from already-formalized theorems. Our contributions are twofold: (1) a novel activation-based intervention for guiding proof synthesis in LLMs; and (2) demonstration that this intervention improves performance under two decoding strategies (sampling and best-first search) without any further training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10895v1">LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 This work has been submitted to IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Medium Access Control (MAC) protocols, essential for wireless networks, are typically manually configured. While deep reinforcement learning (DRL)-based protocols enhance task-specified network performance, they suffer from poor generalizability and resilience, demanding costly retraining to adapt to dynamic environments. To overcome this limitation, we introduce a game-theoretic LLM-empowered multi-agent DRL (MARL) framework, in which the uplink transmission between a base station and a varying number of user equipments is modeled as a dynamic multi-follower Stackelberg game (MFSG), capturing the network's natural hierarchical structure. Within this game, LLM-driven agents, coordinated through proximal policy optimization (PPO), synthesize adaptive, semantic MAC protocols in response to network dynamics. Protocol action grammar (PAG) is employed to ensure the reliability and efficiency of this process. Under this system, we further analyze the existence and convergence behavior in terms of a Stackelberg equilibrium by studying the learning dynamics of LLM-empowered unified policies in response to changing followers. Simulations corroborate that our framework achieves a 77.6% greater throughput and a 65.2% fairness improvement over conventional baselines. Besides, our framework generalizes excellently to a fluctuating number of users without requiring retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07680v3">Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue, but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MoE model, our method speeds up the training by 2.4$\times$ with competitive performance. Codes will be released at https://github.com/ModelTC/HBP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10890v1">LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted by EMNLP2025 System Demonstration
    </div>
    <details class="paper-abstract">
      We introduce LLM x MapReduce-V3, a hierarchically modular agent system designed for long-form survey generation. Building on the prior work, LLM x MapReduce-V2, this version incorporates a multi-agent architecture where individual functional components, such as skeleton initialization, digest construction, and skeleton refinement, are implemented as independent model-context-protocol (MCP) servers. These atomic servers can be aggregated into higher-level servers, creating a hierarchically structured system. A high-level planner agent dynamically orchestrates the workflow by selecting appropriate modules based on their MCP tool descriptions and the execution history. This modular decomposition facilitates human-in-the-loop intervention, affording users greater control and customization over the research process. Through a multi-turn interaction, the system precisely captures the intended research perspectives to generate a comprehensive skeleton, which is then developed into an in-depth survey. Human evaluations demonstrate that our system surpasses representative baselines in both content depth and length, highlighting the strength of MCP-based modular planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v4">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06994v3">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern application codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.19\% (upto 65\%) reduction in application binary size, and an average of 6.08\% (upto 20\%) performance improvement over the traditional profile-guided optimization, without any execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12470v2">Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. In contrast, human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context. This difference between human cognitive flexibility and LLMs' reliance on a single reasoning style raises a critical question: while human fast heuristic reasoning evolved for its efficiency and adaptability, is a uniform reasoning approach truly optimal for LLMs, or does its inflexibility make them brittle and unreliable when faced with tasks demanding more agile, intuitive responses? To answer these questions, we explicitly align LLMs to these reasoning styles by curating a dataset with valid System 1 and System 2 answers, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense reasoning tasks. To analyze the reasoning spectrum, we interpolated between the two extremes by varying the proportion of alignment data, which resulted in a monotonic change in accuracy. A mechanistic analysis of model responses shows that System 1 models employ more definitive outputs, whereas System 2 models demonstrate greater uncertainty. Building on these findings, we further combine System 1- and System 2-aligned models based on the entropy of their generations, without additional training, and obtain a dynamic model that outperforms across nearly all benchmarks. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2208.08067v3">K-ASTRO: Structure-Aware Adaptation of LLMs for Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming software engineering tasks, including code vulnerability detection-a critical area of software security. However, existing methods often rely on resource-intensive models or graph-based techniques, limiting their accessibility and practicality. This paper introduces K-ASTRO, a lightweight Transformer model that combines semantic embeddings from LLMs with structural features of Abstract Syntax Trees (ASTs) to improve both efficiency and accuracy in code vulnerability detection. Our approach introduces an AST-based augmentation technique inspired by mutation testing, a structure-aware attention mechanism that incorporates augmented AST features, and a joint adaptation pipeline to unify code semantics and syntax. Experimental results on three large-scale datasets, including BigVul, DiverseVul, and PrimeVul-demonstrate state-of-the-art performance while enabling rapid inference on CPUs with minimal training time. By offering a scalable, interpretable, and efficient solution, K-ASTRO bridges the gap between LLM advancements and practical software vulnerability detection, providing open-sourced tools to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11974v1">CTIArena: Benchmarking LLM Knowledge and Reasoning Across Heterogeneous Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Under peer-review
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) is central to modern cybersecurity, providing critical insights for detecting and mitigating evolving threats. With the natural language understanding and reasoning capabilities of large language models (LLMs), there is increasing interest in applying them to CTI, which calls for benchmarks that can rigorously evaluate their performance. Several early efforts have studied LLMs on some CTI tasks but remain limited: (i) they adopt only closed-book settings, relying on parametric knowledge without leveraging CTI knowledge bases; (ii) they cover only a narrow set of tasks, lacking a systematic view of the CTI landscape; and (iii) they restrict evaluation to single-source analysis, unlike realistic scenarios that require reasoning across multiple sources. To fill these gaps, we present CTIArena, the first benchmark for evaluating LLM performance on heterogeneous, multi-source CTI under knowledge-augmented settings. CTIArena spans three categories, structured, unstructured, and hybrid, further divided into nine tasks that capture the breadth of CTI analysis in modern security operations. We evaluate ten widely used LLMs and find that most struggle in closed-book setups but show noticeable gains when augmented with security-specific knowledge through our designed retrieval-augmented techniques. These findings highlight the limitations of general-purpose LLMs and the need for domain-tailored techniques to fully unlock their potential for CTI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11967v1">Scaling Long-Horizon LLM Agent via Context-Folding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05670v2">Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v4">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11938v1">FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EuroSys 26
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) in production faces significant challenges from highly variable request patterns and severe resource fragmentation in serverless clusters. Current systems rely on static pipeline configurations that struggle to adapt to dynamic workload conditions, leading to substantial inefficiencies. We present FlexPipe, a novel system that dynamically reconfigures pipeline architectures during runtime to address these fundamental limitations. FlexPipe decomposes models into fine-grained stages and intelligently adjusts pipeline granularity based on real-time request pattern analysis, implementing three key innovations: fine-grained model partitioning with preserved computational graph constraints, inflight pipeline refactoring with consistent cache transitions, and topology-aware resource allocation that navigates GPU fragmentation. Comprehensive evaluation on an 82-GPU cluster demonstrates that FlexPipe achieves up to 8.5x better resource efficiency while maintaining 38.3% lower latency compared to state-of-the-art systems, reducing GPU reservation requirements from 75% to 30% of peak capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11915v1">Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Phishing remains a critical cybersecurity threat, especially with the advent of large language models (LLMs) capable of generating highly convincing malicious content. Unlike earlier phishing attempts which are identifiable by grammatical errors, misspellings, incorrect phrasing, and inconsistent formatting, LLM generated emails are grammatically sound, contextually relevant, and linguistically natural. These advancements make phishing emails increasingly difficult to distinguish from legitimate ones, challenging traditional detection mechanisms. Conventional phishing detection systems often fail when faced with emails crafted by LLMs or manipulated using adversarial perturbation techniques. To address this challenge, we propose a robust phishing email detection system featuring an enhanced text preprocessing pipeline. This pipeline includes spelling correction and word splitting to counteract adversarial modifications and improve detection accuracy. Our approach integrates widely adopted natural language processing (NLP) feature extraction techniques and machine learning algorithms. We evaluate our models on publicly available datasets comprising both phishing and legitimate emails, achieving a detection accuracy of 94.26% and F1-score of 84.39% in model deployment setting. To assess robustness, we further evaluate our models using adversarial phishing samples generated by four attack methods in Python TextAttack framework. Additionally, we evaluate models' performance against phishing emails generated by LLMs including ChatGPT and Llama. Results highlight the resilience of models against evolving AI-powered phishing threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11905v1">LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs) to be reliable, they must learn robust knowledge that can be generally applied in diverse settings -- often unlike those seen during training. Yet, extensive research has shown that LLM performance can be brittle, with models exhibiting excessive sensitivity to trivial input variations. In this work, we explore whether this brittleness is a direct result of unstable internal knowledge representations. To explore this question, we build on previous work showing that LLM representations encode statement truthfulness -- i.e., true, factual statements can be easily separated from false, inaccurate ones. Specifically, we test the robustness of learned knowledge by evaluating representation separability on samples that have undergone superficial transformations to drive them out-of-distribution (OOD), such as typos or reformulations. By applying semantically-preserving perturbations, we study how separability degrades as statements become more OOD, across four LLM families, five evaluation datasets, and three knowledge probing methods. Our results reveal that internal representations of statement truthfulness collapse as the samples' presentations become less similar to those seen during pre-training. While LLMs can often distinguish between true and false statements when they closely resemble the pre-training data, this ability is highly dependent on the statement's exact surface form. These findings offer a possible explanation for brittle benchmark performance: LLMs may learn shallow, non-robust knowledge representations that allow for only limited generalizability. Our work presents a fundamental challenge for the utility of truthfulness probes, and more broadly, calls for further research on improving the robustness of learned knowledge representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20996v2">ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Substance use disorders (SUDs) affect millions of people, and relapses are common, requiring multi-session treatments. Access to care is limited, which contributes to the challenge of recovery support. We present \textbf{ChatThero}, an innovative low-cost, multi-session, stressor-aware, and memory-persistent autonomous \emph{language agent} designed to facilitate long-term behavior change and therapeutic support in addiction recovery. Unlike existing work that mostly finetuned large language models (LLMs) on patient-therapist conversation data, ChatThero was trained in a multi-agent simulated environment that mirrors real therapy. We created anonymized patient profiles from recovery communities (e.g., Reddit). We classify patients as \texttt{easy}, \texttt{medium}, and \texttt{difficult}, three scales representing their resistance to recovery. We created an external environment by introducing stressors (e.g., social determinants of health) to simulate real-world situations. We dynamically inject clinically-grounded therapeutic strategies (motivational interview and cognitive behavioral therapy). Our evaluation, conducted by both human (blinded clinicians) and LLM-as-Judge, shows that ChatThero is superior in empathy and clinical relevance. We show that stressor simulation improves robustness of ChatThero. Explicit stressors increase relapse-like setbacks, matching real-world patterns. We evaluate ChatThero with behavioral change metrics. On a 1--5 scale, ChatThero raises \texttt{motivation} by $+1.71$ points (from $2.39$ to $4.10$) and \texttt{confidence} by $+1.67$ points (from $1.52$ to $3.19$), substantially outperforming GPT-5. On \texttt{difficult} patients, ChatThero reaches the success milestone with $26\%$ fewer turns than GPT-5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11822v1">Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      New Large Language Models (LLMs) become available every few weeks, and modern application developers confronted with the unenviable task of having to decide if they should switch to a new model. While human evaluation remains the gold standard, it is costly and unscalable. The state-of-the-art approach is to use LLMs as evaluators ( LLM-as-a-judge), but this suffers from a critical flaw: LLMs exhibit a strong positive bias. We provide empirical evidence showing that while LLMs can identify valid outputs with high accuracy (i.e., True Positive Rate 96%), they are remarkably poor at identifying invalid ones (i.e., True Negative Rate <25%). This systematic bias, coupled with class imbalance, often leads to inflated reliability scores. While ensemble-based methods like majority voting can help, we show that they are not good enough. We introduce an optimal minority-veto strategy that is resilient to missing data and mitigates this bias to a large extent. For scenarios requiring even higher precision, we propose a novel regression-based framework that directly models the validator bias using a small set of human-annotated ground truth data. On a challenging code feedback task over 366 high-school Python programs, our regression approach reduces the maximum absolute error to just 1.2%, achieving a 2x improvement over the best-performing ensemble of 14 state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11813v1">Task-Aware Reduction for Scalable LLM-Database Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Preprint. Accepted for presentation at the Workshop on Language Models and Databases (LMD), co-located with CASCON 2025 (IEEE). The final version will appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to data-intensive workflows, from database querying to developer observability. Yet the effectiveness of these systems is constrained by the volume, verbosity, and noise of real-world text-rich data such as logs, telemetry, and monitoring streams. Feeding such data directly into LLMs is costly, environmentally unsustainable, and often misaligned with task objectives. Parallel efforts in LLM efficiency have focused on model- or architecture-level optimizations, but the challenge of reducing upstream input verbosity remains underexplored. In this paper, we argue for treating the token budget of an LLM as an attention budget and elevating task-aware text reduction as a first-class design principle for language -- data systems. We position input-side reduction not as compression, but as attention allocation: prioritizing information most relevant to downstream tasks. We outline open research challenges for building benchmarks, designing adaptive reduction pipelines, and integrating token-budget--aware preprocessing into database and retrieval systems. Our vision is to channel scarce attention resources toward meaningful signals in noisy, data-intensive workflows, enabling scalable, accurate, and sustainable LLM--data integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11696v1">QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Code is available at https://github.com/NVlabs/QeRL
    </div>
    <details class="paper-abstract">
      We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for large language models (LLMs). While RL is essential for LLMs' reasoning capabilities, it is resource-intensive, requiring substantial GPU memory and long rollout durations. QeRL addresses these issues by combining NVFP4 quantization with Low-Rank Adaptation (LoRA), accelerating rollout phase of RL while reducing memory overhead. Beyond efficiency, our findings show that quantization noise increases policy entropy, enhancing exploration, and enabling the discovery of better strategies during RL. To further optimize exploration, QeRL introduces an Adaptive Quantization Noise (AQN) mechanism, which dynamically adjusts noise during training. Experiments demonstrate that QeRL delivers over 1.5 times speedup in the rollout phase. Moreover, this is the first framework to enable RL training of a 32B LLM on a single H100 80GB GPU, while delivering overall speedups for RL training. It also achieves faster reward growth and higher final accuracy than 16-bit LoRA and QLoRA, while matching the performance of full-parameter fine-tuning on mathematical benchmarks such as GSM8K (90.8%) and MATH 500 (77.4%) in the 7B model. These results establish QeRL as an efficient and effective framework for RL training in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11695v1">When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v4">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning (NFL-HR)**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the NL-FL Problem Alignment method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the Mixed Problem Input technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based Answer Extraction mechanism. Comprehensive experiments demonstrate that the NFL-HR framework achieves **89.80**% and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by **4.60%** and **4.82%**, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11615v1">LLM-Oriented Token-Adaptive Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) is a key technique for compressing large-scale language models (LLMs), yet prevailing logit-based methods typically employ static strategies that are misaligned with the dynamic learning process of student models. These methods typically treat all tokens indiscriminately and apply a single, fixed temperature, resulting in suboptimal knowledge transfer. To address these limitations, we propose LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD), a novel framework that adapts the distillation process to the real-time learning state of each token. AdaKD consists of two synergistic modules driven by a unified token difficulty metric. First, our Loss-Driven Adaptive Token Focusing (LATF) module dynamically adjusts the distillation focus by monitoring the student's learning stability, concentrating computational resources on the most valuable tokens at each training phase. Second, we introduce Inverse Difficulty Temperature Scaling (IDTS), a counterintuitive yet effective token-level temperature strategy. It employs low temperatures for difficult tokens for targeted error correction, and high temperatures for easy tokens to encourage students to learn from the teacher's complete and smooth output distribution, thereby enhancing generalization. As a plug-and-play framework, AdaKD can consistently improve the performance of various distillation methods on multiple model architectures and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11588v1">Analyzing and Internalizing Complex Policy Documents for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems rely on in-context policy documents encoding diverse business rules. As requirements grow, these documents expand rapidly, causing high computational overhead. This motivates developing internalization methods that embed policy documents into model priors while preserving performance. Prior prompt compression work targets generic prompts, but agentic policy documents span multiple complexity levels and require deeper reasoning, making internalization harder. We introduce CC-Gen, an agentic benchmark generator with Controllable Complexity across four levels, enabling systematic evaluation of agents' ability to handle complexity and offering a unified framework for assessing policy internalization. Our analysis shows that complex policy specifications governing workflows pose major reasoning challenges. Supporting internalization with gold user agent interaction trajectories containing chain-of-thought (CoT) annotations via supervised fine-tuning (SFT) is data-intensive and degrades sharply as policy complexity increases. To mitigate data and reasoning burdens, we propose Category-Aware Policy Continued Pretraining (CAP-CPT). Our automated pipeline parses policy documents to extract key specifications, grouping them into factual, behavioral, and conditional categories, and isolating complex conditions that drive workflow complexity. This guides targeted data synthesis and enables agents to internalize policy information through an autoregressive pretraining loss. Experiments show CAP-CPT improves SFT baselines in all settings, with up to 41% and 22% gains on Qwen-3-32B, achieving 97.3% prompt length reduction on CC-Gen and further enhancing tau-Bench with minimal SFT data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11563v1">Culturally-Aware Conversations: A Framework & Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 To appear at the 4th HCI + NLP Workshop @ EMNLP
    </div>
    <details class="paper-abstract">
      Existing benchmarks that measure cultural adaptation in LLMs are misaligned with the actual challenges these models face when interacting with users from diverse cultural backgrounds. In this work, we introduce the first framework and benchmark designed to evaluate LLMs in realistic, multicultural conversational settings. Grounded in sociocultural theory, our framework formalizes how linguistic style - a key element of cultural communication - is shaped by situational, relational, and cultural context. We construct a benchmark dataset based on this framework, annotated by culturally diverse raters, and propose a new set of desiderata for cross-cultural evaluation in NLP: conversational framing, stylistic sensitivity, and subjective correctness. We evaluate today's top LLMs on our benchmark and show that these models struggle with cultural adaptation in a conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v3">EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07772v2">An approach for systematic decomposition of complex llm tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from reliability issues on complex tasks, as existing decomposition methods are heuristic and rely on agent or manual decomposition. This work introduces a novel, systematic decomposition framework that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leveraging formal complexity measures to guide decomposition. On combinatorial (SATBench) and LLM database querying tasks (Spider), we find that by decomposing the tasks following the measure of complexity, agent can perform considerably better (10-40 percentage point).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11558v1">Zero Data Retention in LLM-based Enterprise AI Assistants: A Comparative Study of Market Leading Agentic AI Products</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Governance of data, compliance, and business privacy matters, particularly for healthcare and finance businesses. Since the recent emergence of AI enterprise AI assistants enhancing business productivity, safeguarding private data and compliance is now a priority. With the implementation of AI assistants across the enterprise, the zero data retention can be achieved by implementing zero data retention policies by Large Language Model businesses like Open AI and Anthropic and Meta. In this work, we explore zero data retention policies for the Enterprise apps of large language models (LLMs). Our key contribution is defining the architectural, compliance, and usability trade-offs of such systems in parallel. In this research work, we examine the development of commercial AI assistants with two industry leaders and market titans in this arena - Salesforce and Microsoft. Both of these companies used distinct technical architecture to support zero data retention policies. Salesforce AgentForce and Microsoft Copilot are among the leading AI assistants providing much-needed push to business productivity in customer care. The purpose of this paper is to analyze the technical architecture and deployment of zero data retention policy by consuming applications as well as big language models service providers like Open Ai, Anthropic, and Meta.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11557v1">Invisible Languages of the LLM Universe</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models are trained on massive multilingual corpora, yet this abundance masks a profound crisis: of the world's 7,613 living languages, approximately 2,000 languages with millions of speakers remain effectively invisible in digital ecosystems. We propose a critical framework connecting empirical measurements of language vitality (real world demographic strength) and digitality (online presence) with postcolonial theory and epistemic injustice to explain why linguistic inequality in AI systems is not incidental but structural. Analyzing data across all documented human languages, we identify four categories: Strongholds (33%, high vitality and digitality), Digital Echoes (6%, high digitality despite declining vitality), Fading Voices (36%, low on both dimensions), and critically, Invisible Giants (27%, high vitality but near-zero digitality) - languages spoken by millions yet absent from the LLM universe. We demonstrate that these patterns reflect continuities from colonial-era linguistic hierarchies to contemporary AI development, constituting what we term digital epistemic injustice. Our analysis reveals that English dominance in AI is not a technical necessity but an artifact of power structures that systematically exclude marginalized linguistic knowledge. We conclude with implications for decolonizing language technology and democratizing access to AI benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11556v1">Personalized and Constructive Feedback for Computer Science Students Using the Large Language Model (LLM)</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The evolving pedagogy paradigms are leading toward educational transformations. One fundamental aspect of effective learning is relevant, immediate, and constructive feedback to students. Providing constructive feedback to large cohorts in academia is an ongoing challenge. Therefore, academics are moving towards automated assessment to provide immediate feedback. However, current approaches are often limited in scope, offering simplistic responses that do not provide students with personalized feedback to guide them toward improvements. This paper addresses this limitation by investigating the performance of Large Language Models (LLMs) in processing students assessments with predefined rubrics and marking criteria to generate personalized feedback for in-depth learning. We aim to leverage the power of existing LLMs for Marking Assessments, Tracking, and Evaluation (LLM-MATE) with personalized feedback to enhance students learning. To evaluate the performance of LLM-MATE, we consider the Software Architecture (SA) module as a case study. The LLM-MATE approach can help module leaders overcome assessment challenges with large cohorts. Also, it helps students improve their learning by obtaining personalized feedback in a timely manner. Additionally, the proposed approach will facilitate the establishment of ground truth for automating the generation of students assessment feedback using the ChatGPT API, thereby reducing the overhead associated with large cohort assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v3">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 30 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v2">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11498v1">ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at algorithmic code generation, they struggle with front-end development, where correctness is judged on rendered pixels and interaction. We present ReLook, an agentic, vision-grounded reinforcement learning framework that empowers an agent to close a robust generate--diagnose--refine loop by invoking a multimodal LLM (MLLM) as a tool. During training, the agent uses the MLLM-in-the-loop both as a visual critic--scoring code with screenshots--and as a source of actionable, vision-grounded feedback; a strict zero-reward rule for invalid renders anchors renderability and prevents reward hacking. To prevent behavioral collapse, we introduce Forced Optimization, a strict acceptance rule that admits only improving revisions, yielding monotonically better trajectories. At inference, we decouple the critic and run a lightweight, critic-free self-edit cycle, keeping latency comparable to base decoding while retaining most of the gains. Across three widely used benchmarks, ReLook consistently outperforms strong baselines in vision-grounded front-end code generation, highlighting the benefits of agentic perception, visual rewards, and training-inference decoupling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04641v2">Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02502v3">LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 ACL 2025, our code is available at https://github.com/ZNLP/LADM
    </div>
    <details class="paper-abstract">
      Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05605v4">Evolving LLMs' Self-Refinement Capability via Synergistic Training-Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Self-Refinement refers to a model's ability to revise its own responses to produce improved outputs. This capability can also serve as a fundamental mechanism for Self-Improvement, for example, by reconstructing datasets with refined results to enhance intrinsic model performance. However, our comprehensive experiments reveal that large language models (LLMs) show no clear evidence of inherent Self-Refinement and may even experience response quality degradation after Self-Refinement. To address this issue, we propose EVOLVE, a simple and effective framework for eliciting and tracking the evolution of Self-Refinement through iterative training. We first explore optimization methods during training to activate the model's Self-Refinement capability. Then, at inference, we investigate various generation strategies to further enhance and utilize Self-Refinement while supplying the necessary data for training. Through synergistic optimization of training and inference stages, we continually evolve the model's Self-Refinement ability, enabling it to better refine its own responses. Moreover, we demonstrate the potential of leveraging Self-Refinement to achieve broader Self-Improvement of intrinsic model abilities. Experiments show that the evolved Self-Refinement ability enables the Llama-3.1-8B base model to surpass GPT-4o, achieving 62.3% length-controlled and 63.3% raw win rates on AlpacaEval 2, and 50.3% on Arena-Hard. It also generalizes effectively to out-of-domain reasoning tasks, improving performance on mathematical reasoning benchmarks such as GSM8K and MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11434v1">Who are you, ChatGPT? Personality and Demographic Style in LLM-Generated Content</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 ECAI2025 (Identity-Aware AI workshop)
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have become central to everyday life, producing human-like text across diverse domains. A growing body of research investigates whether these models also exhibit personality- and demographic-like characteristics in their language. In this work, we introduce a novel, data-driven methodology for assessing LLM personality without relying on self-report questionnaires, applying instead automatic personality and gender classifiers to model replies on open-ended questions collected from Reddit. Comparing six widely used models to human-authored responses, we find that LLMs systematically express higher Agreeableness and lower Neuroticism, reflecting cooperative and stable conversational tendencies. Gendered language patterns in model text broadly resemble those of human writers, though with reduced variation, echoing prior findings on automated agents. We contribute a new dataset of human and model responses, along with large-scale comparative analyses, shedding new light on the topic of personality and demographic patterns of generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11423v1">Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), enables users to flag misleading posts, attach contextual notes, and vote on their helpfulness. However, our analysis of 30.8K health-related notes reveals significant latency, with a median delay of 17.6 hours before the first note receives a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified framework that leverages large language models (LLMs) to augment Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two complementary modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, along with a hierarchical three-step evaluation that progressively assesses relevance, correctness, and helpfulness. We instantiate the framework through HealthNotes, a benchmark of 1.2K helpfulness-annotated health notes paired with a fine-tuned helpfulness judge. Experiments on fifteen LLMs reveal an overlooked loophole in current helpfulness evaluation, where stylistic fluency is mistaken for factual accuracy, and demonstrate that our hierarchical evaluation and LLM-augmented generation jointly enhance factual precision and evidence utility. These results point toward a hybrid human-AI governance model that improves both the rigor and timeliness of crowd-sourced fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11414v1">Uncertainty-Aware, Risk-Adaptive Access Control for Agentic Systems using an LLM-Judged TBAC Model</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The proliferation of autonomous AI agents within enterprise environments introduces a critical security challenge: managing access control for emergent, novel tasks for which no predefined policies exist. This paper introduces an advanced security framework that extends the Task-Based Access Control (TBAC) model by using a Large Language Model (LLM) as an autonomous, risk-aware judge. This model makes access control decisions not only based on an agent's intent but also by explicitly considering the inherent \textbf{risk associated with target resources} and the LLM's own \textbf{model uncertainty} in its decision-making process. When an agent proposes a novel task, the LLM judge synthesizes a just-in-time policy while also computing a composite risk score for the task and an uncertainty estimate for its own reasoning. High-risk or high-uncertainty requests trigger more stringent controls, such as requiring human approval. This dual consideration of external risk and internal confidence allows the model to enforce a more robust and adaptive version of the principle of least privilege, paving the way for safer and more trustworthy autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11409v1">Leveraging LLMs for Semi-Automatic Corpus Filtration in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      The creation of systematic literature reviews (SLR) is critical for analyzing the landscape of a research field and guiding future research directions. However, retrieving and filtering the literature corpus for an SLR is highly time-consuming and requires extensive manual effort, as keyword-based searches in digital libraries often return numerous irrelevant publications. In this work, we propose a pipeline leveraging multiple large language models (LLMs), classifying papers based on descriptive prompts and deciding jointly using a consensus scheme. The entire process is human-supervised and interactively controlled via our open-source visual analytics web interface, LLMSurver, which enables real-time inspection and modification of model outputs. We evaluate our approach using ground-truth data from a recent SLR comprising over 8,000 candidate papers, benchmarking both open and commercial state-of-the-art LLMs from mid-2024 and fall 2025. Results demonstrate that our pipeline significantly reduces manual effort while achieving lower error rates than single human annotators. Furthermore, modern open-source models prove sufficient for this task, making the method accessible and cost-effective. Overall, our work demonstrates how responsible human-AI collaboration can accelerate and enhance systematic literature reviews within academic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11398v1">Living Off the LLM: How LLMs Will Change Adversary Tactics</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 6 pages, 0 figures
    </div>
    <details class="paper-abstract">
      In living off the land attacks, malicious actors use legitimate tools and processes already present on a system to avoid detection. In this paper, we explore how the on-device LLMs of the future will become a security concern as threat actors integrate LLMs into their living off the land attack pipeline and ways the security community may mitigate this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11389v1">Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 34 pages, 32figures
    </div>
    <details class="paper-abstract">
      Social deduction games like Werewolf combine language, reasoning, and strategy, providing a testbed for studying natural language and social intelligence. However, most studies reduce the game to LLM-based self-play, yielding templated utterances and anecdotal cases that overlook the richness of social gameplay. Evaluation further relies on coarse metrics such as survival time or subjective scoring due to the lack of quality reference data. To address these gaps, we curate a high-quality, human-verified multimodal Werewolf dataset containing over 100 hours of video, 32.4M utterance tokens, and 15 rule variants. Based on this dataset, we propose a novel strategy-alignment evaluation that leverages the winning faction's strategies as ground truth in two stages: 1) Speech evaluation, formulated as multiple-choice-style tasks that assess whether the model can adopt appropriate stances across five dimensions of social ability; and 2) Decision evaluation, which assesses the model's voting choices and opponent-role inferences. This framework enables a fine-grained evaluation of models' linguistic and reasoning capabilities, while capturing their ability to generate strategically coherent gameplay. Our experiments show that state-of-the-art LLMs show diverse performance, with roughly half remain below 0.50, revealing clear gaps in deception and counterfactual reasoning. We hope our dataset further inspires research on language, reasoning, and strategy in multi-agent interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25123v2">From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11358v1">LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. While traditional retrieval focuses on relevance, RAG's effectiveness depends on the utility of retrieved passages, i.e., the usefulness in facilitating the generation of an accurate and comprehensive answer. Existing studies often treat utility as a generic attribute, ignoring the fact that different LLMs may benefit differently from the same passage due to variations in internal knowledge and comprehension ability. In this work, we introduce and systematically investigate the notion of LLM-specific utility. Through large-scale experiments across multiple datasets and LLMs, we demonstrate that human-annotated passages are not optimal for LLMs and that ground-truth utilitarian passages are not transferable across different LLMs. These findings highlight the necessity of adopting the LLM-specific utility in RAG research. Our findings indicate that some human-annotated passages are not ground-truth utilitarian passages for specific LLMs, partially due to the varying readability of queries and passages for LLMs, a tendency for which perplexity is a key metric. Based on these findings, we propose a benchmarking procedure for LLM-specific utility judgments. We evaluate existing utility judgment methods on six datasets and find that while verbalized methods using pseudo-answers perform robustly, LLMs struggle to assess utility effectively-failing to reject all passages for known queries and to select truly useful ones for unknown queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00761v3">Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often fragile: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the optimizer, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the 'grade' of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a hybrid optimizer that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11331v1">Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference at the network edge is a promising serving paradigm that leverages distributed edge resources to run inference near users and enhance privacy. Existing edge-based LLM inference systems typically adopt autoregressive decoding (AD), which only generates one token per forward pass. This iterative process, compounded by the limited computational resources of edge nodes, results in high serving latency and constrains the system's ability to support multiple users under growing demands.To address these challenges, we propose a speculative decoding (SD)-based LLM serving framework that deploys small and large models across heterogeneous edge nodes to collaboratively deliver inference services. Specifically, the small model rapidly generates draft tokens that the large model verifies in parallel, enabling multi-token generation per forward pass and thus reducing serving latency. To improve resource utilization of edge nodes, we incorporate pipeline parallelism to overlap drafting and verification across multiple inference tasks. Based on this framework, we analyze and derive a comprehensive latency model incorporating both communication and inference latency. Then, we formulate a joint optimization problem for speculation length, task batching, and wireless communication resource allocation to minimize total serving latency. To address this problem, we derive the closed-form solutions for wireless communication resource allocation, and develop a dynamic programming algorithm for joint batching and speculation control strategies. Experimental results demonstrate that the proposed framework achieves lower serving latency compared to AD-based serving systems. In addition,the proposed joint optimization method delivers up to 44.9% latency reduction compared to benchmark schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11328v1">Do LLMs "Feel"? Emotion Circuits Discovery and Control</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 19 pages, 8 figures, 8 tables. Code and dataset available at https://github.com/Aurora-cx/EmotionCircuits-LLM
    </div>
    <details class="paper-abstract">
      As the demand for emotional intelligence in large language models (LLMs) grows, a key challenge lies in understanding the internal mechanisms that give rise to emotional expression and in controlling emotions in generated text. This study addresses three core questions: (1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control? We first construct a controlled dataset, SEV (Scenario-Event with Valence), to elicit comparable internal states across emotions. Subsequently, we extract context-agnostic emotion directions that reveal consistent, cross-context encoding of emotion (Q1). We identify neurons and attention heads that locally implement emotional computation through analytical decomposition and causal analysis, and validate their causal roles via ablation and enhancement interventions. Next, we quantify each sublayer's causal influence on the model's final emotion representation and integrate the identified local components into coherent global emotion circuits that drive emotional expression (Q2). Directly modulating these circuits achieves 99.65% emotion-expression accuracy on the test set, surpassing prompting- and steering-based methods (Q3). To our knowledge, this is the first systematic study to uncover and validate emotion circuits in LLMs, offering new insights into interpretability and controllable emotional intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11313v1">Automated Skill Decomposition Meets Expert Ontologies: Bridging the Granularity Gap with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      This paper investigates automated skill decomposition using Large Language Models (LLMs) and proposes a rigorous, ontology-grounded evaluation framework. Our framework standardizes the pipeline from prompting and generation to normalization and alignment with ontology nodes. To evaluate outputs, we introduce two metrics: a semantic F1-score that uses optimal embedding-based matching to assess content accuracy, and a hierarchy-aware F1-score that credits structurally correct placements to assess granularity. We conduct experiments on ROME-ESCO-DecompSkill, a curated subset of parents, comparing two prompting strategies: zero-shot and leakage-safe few-shot with exemplars. Across diverse LLMs, zero-shot offers a strong baseline, while few-shot consistently stabilizes phrasing and granularity and improves hierarchy-aware alignment. A latency analysis further shows that exemplar-guided prompts are competitive - and sometimes faster - than unguided zero-shot due to more schema-compliant completions. Together, the framework, benchmark, and metrics provide a reproducible foundation for developing ontology-faithful skill decomposition systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11288v1">Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across three datasets, three frontier models produce broadly misaligned responses at rates between 2% and 17% given 64 narrow in-context examples, and up to 58% with 256 examples. We also examine mechanisms of EM by eliciting step-by-step reasoning (while leaving in-context examples unchanged). Manual analysis of the resulting chain-of-thought shows that 67.5% of misaligned traces explicitly rationalize harmful outputs by adopting a reckless or dangerous ''persona'', echoing prior results on finetuning-induced EM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11282v1">Vision-LLMs for Spatiotemporal Traffic Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Accurate spatiotemporal traffic forecasting is a critical prerequisite for proactive resource management in dense urban mobile networks. While Large Language Models (LLMs) have shown promise in time series analysis, they inherently struggle to model the complex spatial dependencies of grid-based traffic data. Effectively extending LLMs to this domain is challenging, as representing the vast amount of information from dense geographical grids can be inefficient and overwhelm the model's context. To address these challenges, we propose ST-Vision-LLM, a novel framework that reframes spatiotemporal forecasting as a vision-language fusion problem. Our approach leverages a Vision-LLM visual encoder to process historical global traffic matrices as image sequences, providing the model with a comprehensive global view to inform cell-level predictions. To overcome the inefficiency of LLMs in handling numerical data, we introduce an efficient encoding scheme that represents floating-point values as single tokens via a specialized vocabulary, coupled with a two-stage numerical alignment fine-tuning process. The model is first trained with Supervised Fine-Tuning (SFT) and then further optimized for predictive accuracy using Group Relative Policy Optimization (GRPO), a memory-efficient reinforcement learning method. Evaluations on real-world mobile traffic datasets demonstrate that ST-Vision-LLM outperforms existing methods by 15.6% in long-term prediction accuracy and exceeds the second-best baseline by over 30.04% in cross-domain few-shot scenarios. Our extensive experiments validate the model's strong generalization capabilities across various data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v3">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 10 pages, 13 tables, 14 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for non-verifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v3">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4\% absolute on SAT MATH). Furthermore, we benchmark our PRM against existing open-source reward models, demonstrating superior alignment with reasoning quality and more consistent guidance for downstream generation. Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11246v1">Collaborative Shadows: Distributed Backdoor Attacks in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) demonstrate increasing integration into next-generation applications, but their safety in backdoor attacks remains largely underexplored. However, existing research has focused exclusively on single-agent backdoor attacks, overlooking the novel attack surfaces introduced by agent collaboration in MAS. To bridge this gap, we present the first Distributed Backdoor Attack tailored to MAS. We decompose the backdoor into multiple distributed attack primitives that are embedded within MAS tools. These primitives remain dormant individually but collectively activate only when agents collaborate in a specific sequence, thereby assembling the full backdoor to execute targeted attacks such as data exfiltration. To fully assess this threat, we introduce a benchmark for multi-role collaborative tasks and a sandboxed framework to evaluate. Extensive experiments demonstrate that our attack achieves an attack success rate exceeding 95% without degrading performance on benign tasks. This work exposes novel backdoor attack surfaces that exploit agent collaboration, underscoring the need to move beyond single-agent protection. Code and benchmark are available at https://github.com/whfeLingYu/Distributed-Backdoor-Attacks-in-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11218v1">The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v2">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11203v1">TraceAegis: Securing LLM-Based Agents via Hierarchical and Behavioral Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      LLM-based agents have demonstrated promising adaptability in real-world applications. However, these agents remain vulnerable to a wide range of attacks, such as tool poisoning and malicious instructions, that compromise their execution flow and can lead to serious consequences like data breaches and financial loss. Existing studies typically attempt to mitigate such anomalies by predefining specific rules and enforcing them at runtime to enhance safety. Yet, designing comprehensive rules is difficult, requiring extensive manual effort and still leaving gaps that result in false negatives. As agent systems evolve into complex software systems, we take inspiration from software system security and propose TraceAegis, a provenance-based analysis framework that leverages agent execution traces to detect potential anomalies. In particular, TraceAegis constructs a hierarchical structure to abstract stable execution units that characterize normal agent behaviors. These units are then summarized into constrained behavioral rules that specify the conditions necessary to complete a task. By validating execution traces against both hierarchical and behavioral constraints, TraceAegis is able to effectively detect abnormal behaviors. To evaluate the effectiveness of TraceAegis, we introduce TraceAegis-Bench, a dataset covering two representative scenarios: healthcare and corporate procurement. Each scenario includes 1,300 benign behaviors and 300 abnormal behaviors, where the anomalies either violate the agent's execution order or break the semantic consistency of its execution sequence. Experimental results demonstrate that TraceAegis achieves strong performance on TraceAegis-Bench, successfully identifying the majority of abnormal behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v4">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Accepted as a main paper at EMNLP 2025. CR version
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics for literature prioritize mechanical accuracy over artistic expression and tend to overrate machine translation as being superior to human translation from experienced professionals. In the long run, this bias could result in an irreversible decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LITRANSPROQA, a novel, reference-free, LLM-based question-answering framework designed for literary translation evaluation. LITRANSPROQA integrates humans in the loop to incorporate insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LITRANSPROQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LITRANSPROQA reaches an adequacy performance comparable to trained linguistic student evaluators, though it still falls behind experienced professional translators. LITRANSPROQA shows broad applicability to open-source models like LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free tool for evaluating literary translations that require local processing due to copyright or ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11192v1">Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 8 pages, to appear in IEEE Cross-disciplinary Conference on Memory-Centric Computing (CCMCC)
    </div>
    <details class="paper-abstract">
      Structured sparsity enables deploying large language models (LLMs) on resource-constrained systems. Approaches like dense-to-sparse fine-tuning are particularly compelling, achieving remarkable structured sparsity by reducing the model size by over 6.7x, while still maintaining acceptable accuracy. Despite this reduction, LLM inference, especially the decode stage being inherently memory-bound, is extremely expensive on conventional Von-Neumann architectures. Compute-in-memory (CIM) architectures mitigate this by performing computations directly in memory, and when paired with sparse LLMs, enable storing and computing the entire model in memory, eliminating the data movement on the off-chip bus and improving efficiency. Nonetheless, naively mapping sparse matrices onto CIM arrays leads to poor array utilization and diminished computational efficiency. In this paper, we present an automated framework with novel mapping and scheduling strategies to accelerate sparse LLM inference on CIM accelerators. By exploiting block-diagonal sparsity, our approach improves CIM array utilization by over 50%, achieving more than 4x reduction in both memory footprint and the number of required floating-point operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11188v1">Protein as a Second Language for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 Main paper: 9 pages, 6 figures. With references and appendix: 18 pages, 9 figures total. Submitted to ICLR 2026 (under review)
    </div>
    <details class="paper-abstract">
      Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the "Protein-as-Second-Language" framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence-question-answer triples that reveal functional cues in a zero-shot setting, without any further training. To support this process, we curate a bilingual corpus of 79,926 protein-QA instances spanning attribute prediction, descriptive understanding, and extended reasoning. Empirically, our method delivers consistent gains across diverse open-source LLMs and GPT-4, achieving up to 17.2% ROUGE-L improvement (average +7%) and even surpassing fine-tuned protein-specific language models. These results highlight that generic LLMs, when guided with protein-as-language cues, can outperform domain-specialized models, offering a scalable pathway for protein understanding in foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04651v2">Agents of Change: Self-Evolving LLM Agents for Strategic Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      We address the long-horizon gap in large language model (LLM) agents by enabling them to sustain coherent strategies in adversarial, stochastic environments. Settlers of Catan provides a challenging benchmark: success depends on balancing short- and long-term goals amid randomness, trading, expansion, and blocking. Prompt-centric LLM agents (e.g., ReAct, Reflexion) must re-interpret large, evolving game states each turn, quickly saturating context windows and losing strategic consistency. We propose HexMachina, a continual learning multi-agent system that separates environment discovery (inducing an adapter layer without documentation) from strategy improvement (evolving a compiled player through code refinement and simulation). This design preserves executable artifacts, allowing the LLM to focus on high-level strategy rather than per-turn reasoning. In controlled Catanatron experiments, HexMachina learns from scratch and evolves players that outperform the strongest human-crafted baseline (AlphaBeta), achieving a 54% win rate and surpassing prompt-driven and no-discovery baselines. Ablations confirm that isolating pure strategy learning improves performance. Overall, artifact-centric continual learning transforms LLMs from brittle stepwise deciders into stable strategy designers, advancing long-horizon autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11151v1">TypePilot: Leveraging the Scala Type System for Secure LLM-generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Large language Models (LLMs) have shown remarkable proficiency in code generation tasks across various programming languages. However, their outputs often contain subtle but critical vulnerabilities, posing significant risks when deployed in security-sensitive or mission-critical systems. This paper introduces TypePilot, an agentic AI framework designed to enhance the security and robustness of LLM-generated code by leveraging strongly typed and verifiable languages, using Scala as a representative example. We evaluate the effectiveness of our approach in two settings: formal verification with the Stainless framework and general-purpose secure code generation. Our experiments with leading open-source LLMs reveal that while direct code generation often fails to enforce safety constraints, just as naive prompting for more secure code, our type-focused agentic pipeline substantially mitigates input validation and injection vulnerabilities. The results demonstrate the potential of structured, type-guided LLM workflows to improve the SotA of the trustworthiness of automated code generation in high-assurance domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v2">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11129v1">video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio-visual LLM that, to our knowledge, is the first to process 3-hour videos at 1 FPS and 360p resolution under a fixed memory budget. Our model introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT_HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN S sustains high-quality understanding on multi-hour videos with 10k frames and 1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, outperforming both offline and streaming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11121v1">Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used as automated heuristic designers for vehicle routing problems (VRPs), current state-of-the-art methods predominantly rely on prompting massive, general-purpose models like GPT-4. This work challenges that paradigm by demonstrating that a smaller, specialized LLM, when meticulously fine-tuned, can generate components that surpass expert-crafted heuristics within advanced solvers. We propose RFTHGS, a novel Reinforcement learning (RL) framework for Fine-Tuning a small LLM to generate high-performance crossover operators for the Hybrid Genetic Search (HGS) solver, applied to the Capacitated VRP (CVRP). Our method employs a multi-tiered, curriculum-based reward function that progressively guides the LLM to master generating first compilable, then executable, and finally, superior-performing operators that exceed human expert designs. This is coupled with an operator caching mechanism that discourages plagiarism and promotes diversity during training. Comprehensive experiments show that our fine-tuned LLM produces crossover operators which significantly outperform the expert-designed ones in HGS. The performance advantage remains consistent, generalizing from small-scale instances to large-scale problems with up to 1000 nodes. Furthermore, RFTHGS exceeds the performance of leading neuro-combinatorial baselines, prompt-based methods, and commercial LLMs such as GPT-4o and GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11108v1">A Vision for Access Control in LLM-based Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12421v2">Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted by NeurIPS 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Unlike reasoning, which often entails a deep sequence of deductive steps, complex real-world planning is characterized by the need to synthesize a broad spectrum of parallel and potentially conflicting information and constraints. For example, in travel planning scenarios, it requires the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints, leading to suboptimal solutions. Motivated by the challenges of real-world travel planning, this paper introduces the Multiple Aspects of Planning (MAoP), empowering LLMs with "wide-horizon thinking" to solve planning problems with multifaceted constraints. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planners, enabling strong inference-time scalability by scaling aspects to consider various constraints. In addition, existing benchmarks for multi-constraint planning are flawed because they assess constraints in isolation, ignoring causal dependencies within the constraints, e.g, travel planning, where past activities dictate future itinerary. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world simulation, thereby inherently resolving these causal dependencies. This paper advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10592v1">A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving.
    </details>
</div>
