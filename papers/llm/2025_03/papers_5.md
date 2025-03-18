# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00053v3">ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated a remarkable ability to serve as general-purpose tools for various language-based tasks. Recent works have demonstrated that the efficacy of such models can be improved through iterative dialog between multiple models. While these paradigms show promise in improving model efficacy, most works in this area treat collaboration as an emergent behavior, rather than a learned behavior. In doing so, current multi-agent frameworks rely on collaborative behaviors to have been sufficiently trained into off-the-shelf models. To address this limitation, we propose ACC-Collab, an Actor-Critic based learning framework to produce a two-agent team (an actor-agent and a critic-agent) specialized in collaboration. We demonstrate that ACC-Collab outperforms SotA multi-agent techniques on a wide array of benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00153v2">Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Probing learned concepts in large language models (LLMs) is crucial for understanding how semantic knowledge is encoded internally. Training linear classifiers on probing tasks is a principle approach to denote the vector of a certain concept in the representation space. However, the single vector identified for a concept varies with both data and training, making it less robust and weakening its effectiveness in real-world applications. To address this challenge, we propose an approach to approximate the subspace representing a specific concept. Built on linear probing classifiers, we extend the concept vectors into Gaussian Concept Subspace (GCS). We demonstrate GCS's effectiveness through measuring its faithfulness and plausibility across multiple LLMs with different sizes and architectures. Additionally, we use representation intervention tasks to showcase its efficacy in real-world applications such as emotion steering. Experimental results indicate that GCS concept vectors have the potential to balance steering performance and maintaining the fluency in natural language generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09990v2">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04554v1">Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The ability of generative large language models (LLMs) to perform in-context learning has given rise to a large body of research into how best to prompt models for various natural language processing tasks. Machine Translation (MT) has been shown to benefit from in-context examples, in particular when they are semantically similar to the sentence to translate. In this paper, we propose a new LLM-based translation paradigm, compositional translation, to replace naive few-shot MT with similarity-based demonstrations. An LLM is used to decompose a sentence into simpler phrases, and then to translate each phrase with the help of retrieved demonstrations. Finally, the LLM is prompted to translate the initial sentence with the help of the self-generated phrase-translation pairs. Our intuition is that this approach should improve translation because these shorter phrases should be intrinsically easier to translate and easier to match with relevant examples. This is especially beneficial in low-resource scenarios, and more generally whenever the selection pool is small or out of domain. We show that compositional translation boosts LLM translation performance on a wide range of popular MT benchmarks, including FLORES 200, NTREX 128 and TICO-19. Code and outputs are available at https://github.com/ArmelRandy/compositional-translation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20984v2">UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at https://github.com/tongwu17/SemEval-2025-Task1-UoR-NCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04474v1">Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to the ICBINB Workshop at ICLR'25
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04463v1">Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The need for interpretability in deep learning has driven interest in counterfactual explanations, which identify minimal changes to an instance that change a model's prediction. Current counterfactual (CF) generation methods require task-specific fine-tuning and produce low-quality text. Large Language Models (LLMs), though effective for high-quality text generation, struggle with label-flipping counterfactuals (i.e., counterfactuals that change the prediction) without fine-tuning. We introduce two simple classifier-guided approaches to support counterfactual generation by LLMs, eliminating the need for fine-tuning while preserving the strengths of LLMs. Despite their simplicity, our methods outperform state-of-the-art counterfactual generation methods and are effective across different LLMs, highlighting the benefits of guiding counterfactual generation by LLMs with classifier information. We further show that data augmentation by our generated CFs can improve a classifier's robustness. Our analysis reveals a critical issue in counterfactual generation by LLMs: LLMs rely on parametric knowledge rather than faithfully following the classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v1">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 To appear at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose $\textit{Adaptive Branching Monte Carlo Tree Search (AB-MCTS)}$, a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04381v1">TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Codes and models are available at https://github.com/d223302/TRACT
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm uses large language models (LLMs) for automated text evaluation, where a numerical assessment is assigned by an LLM to the input text following scoring rubrics. Existing methods for LLM-as-a-judge use cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of score prediction. Recent work addresses numerical prediction limitations of LLM fine-tuning through regression-aware fine-tuning, which, however, does not consider chain-of-thought (CoT) reasoning for score prediction. In this paper, we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method combining CoT reasoning with regression-aware training. TRACT consists of two stages: first, seed LLM is fine-tuned to generate CoTs, which serve as supervision for the second stage fine-tuning. The training objective of TRACT combines the CE loss for learning the CoT reasoning capabilities, and the regression-aware loss for the score prediction. Experiments across four LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms existing methods. Extensive ablation studies validate the importance of each component in TRACT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04377v1">How can representation dimension dominate structurally pruned LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ICLR 2025 Workshop on Sparsity in LLMs (SLLM)
    </div>
    <details class="paper-abstract">
      Pruning assumes a subnetwork exists in the original deep neural network, which can achieve comparative model performance with less computation than the original. However, it is unclear how the model performance varies with the different subnetwork extractions. In this paper, we choose the representation dimension (or embedding dimension, model dimension, the dimension of the residual stream in the relevant literature) as the entry point to this issue. We investigate the linear transformations in the LLM transformer blocks and consider a specific structured pruning approach, SliceGPT, to extract the subnetworks of different representation dimensions. We mechanistically analyse the activation flow during the model forward passes, and find the representation dimension dominates the linear transformations, model predictions, and, finally, the model performance. Explicit analytical relations are given to calculate the pruned model performance (perplexity and accuracy) without actual evaluation, and are empirically validated with Llama-3-8B-Instruct and Phi-3-mini-4k-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04369v1">Lost in Literalism: How Supervised Training Shapes Translationese in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 19 pages;
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in machine translation, demonstrating impressive performance across diverse languages. However, translationese, characterized by overly literal and unnatural translations, remains a persistent challenge in LLM-based translation systems. Despite their pre-training on vast corpora of natural utterances, LLMs exhibit translationese errors and generate unexpected unnatural translations, stemming from biases introduced during supervised fine-tuning (SFT). In this work, we systematically evaluate the prevalence of translationese in LLM-generated translations and investigate its roots during supervised training. We introduce methods to mitigate these biases, including polishing golden references and filtering unnatural training instances. Empirical evaluations demonstrate that these approaches significantly reduce translationese while improving translation naturalness, validated by human evaluations and automatic metrics. Our findings highlight the need for training-aware adjustments to optimize LLM translation outputs, paving the way for more fluent and target-language-consistent translations. We release the data and code at https://github.com/yafuly/LLM_Translationese.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04360v1">Exploring the Multilingual NLG Evaluation Abilities of LLM-Based Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Previous research has shown that LLMs have potential in multilingual NLG evaluation tasks. However, existing research has not fully explored the differences in the evaluation capabilities of LLMs across different languages. To this end, this study provides a comprehensive analysis of the multilingual evaluation performance of 10 recent LLMs, spanning high-resource and low-resource languages through correlation analysis, perturbation attacks, and fine-tuning. We found that 1) excluding the reference answer from the prompt and using large-parameter LLM-based evaluators leads to better performance across various languages; 2) most LLM-based evaluators show a higher correlation with human judgments in high-resource languages than in low-resource languages; 3) in the languages where they are most sensitive to such attacks, they also tend to exhibit the highest correlation with human judgments; and 4) fine-tuning with data from a particular language yields a broadly consistent enhancement in the model's evaluation performance across diverse languages. Our findings highlight the imbalance in LLMs'evaluation capabilities across different languages and suggest that low-resource language scenarios deserve more attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04302v1">Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04291v1">MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Published in AAAI 2025
    </div>
    <details class="paper-abstract">
      We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04271v1">On Fact and Frequency: LLM Responses to Misinformation Expressed with Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 4 pages, 1 figure, 3 tables, conference
    </div>
    <details class="paper-abstract">
      We study LLM judgments of misinformation expressed with uncertainty. Our experiments study the response of three widely used LLMs (GPT-4o, LlaMA3, DeepSeek-v2) to misinformation propositions that have been verified false and then are transformed into uncertain statements according to an uncertainty typology. Our results show that after transformation, LLMs change their factchecking classification from false to not-false in 25% of the cases. Analysis reveals that the change cannot be explained by predictors to which humans are expected to be sensitive, i.e., modality, linguistic cues, or argumentation strategy. The exception is doxastic transformations, which use linguistic cue phrases such as "It is believed ...".To gain further insight, we prompt the LLM to make another judgment about the transformed misinformation statements that is not related to truth value. Specifically, we study LLM estimates of the frequency with which people make the uncertain statement. We find a small but significant correlation between judgment of fact and estimation of frequency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04253v1">ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 11pages, 17 figures
    </div>
    <details class="paper-abstract">
      The growing adoption of Large Language Models (LLMs) across various domains has driven the demand for efficient and scalable AI-serving solutions. Deploying LLMs requires optimizations to manage their significant computational and data demands. The prefill stage processes large numbers of input tokens in parallel, increasing computational load, while the decoding stage relies heavily on memory bandwidth due to the auto-regressive nature of LLMs. Current hardware, such as GPUs, often fails to balance these demands, leading to inefficient utilization. While batching improves hardware efficiency, it delays response times, degrading Quality-of-Service (QoS). This disconnect between vendors, who aim to maximize resource efficiency, and users, who prioritize low latency, highlights the need for a better solution. To address this, we propose ADOR, a framework that automatically identifies and recommends hardware architectures tailored to LLM serving. By leveraging predefined architecture templates specialized for heterogeneous dataflows, ADOR optimally balances throughput and latency. It efficiently explores design spaces to suggest architectures that meet the requirements of both vendors and users. ADOR demonstrates substantial performance improvements, achieving 2.51x higher QoS and 4.01x better area efficiency compared to the A100 at high batch sizes, making it a robust solution for scalable and cost-effective LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04241v1">ThrowBench: Benchmarking LLMs by Predicting Runtime Exceptions</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) have shown astounding capabilities of code understanding and synthesis. In order to assess such capabilities, several benchmarks have been devised (e.g., HumanEval). However, most benchmarks focus on code synthesis from natural language instructions. Hence, such benchmarks do not test for other forms of code understanding. Moreover, there have been concerns about contamination and leakage. That is, benchmark problems (or closely related problems) may appear in training set, strongly biasing benchmark results. In this work we investigate whether large language models can correctly predict runtime program behavior. To this end, we introduce ThrowBench, a benchmark consisting of over 2,400 short user-written programs written in four different programming languages. The majority of these programs throw an exception during runtime (due to a bug). LLMs are asked to predict whether a presented program throws an exception and, if so, which one. Evaluating our benchmark on six state-of-the-art code LLMs we see modest performance ranging from 19 to 38% (F1 score). Benchmarking a wider set of code capabilities could improve the assessment of code LLMs and help identify weak points in current models. Moreover, as ground-truth answers have been determined through program execution, leakage is not a concern. We release ThrowBench as well as all of our results together with this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13681v2">An LLM-based Agent for Reliable Docker Environment Configuration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Environment configuration is a critical yet time-consuming step in software development, especially when dealing with unfamiliar code repositories. While Large Language Models (LLMs) demonstrate the potential to accomplish software engineering tasks, existing methods for environment configuration often rely on manual efforts or fragile scripts, leading to inefficiencies and unreliable outcomes. We introduce Repo2Run, the first LLM-based agent designed to fully automate environment configuration and generate executable Dockerfiles for arbitrary Python repositories. We address two major challenges: (1) enabling the LLM agent to configure environments within isolated Docker containers, and (2) ensuring the successful configuration process is recorded and accurately transferred to a Dockerfile without error. To achieve this, we propose atomic configuration synthesis, featuring a dual-environment architecture (internal and external environment) with a rollback mechanism to prevent environment "pollution" from failed commands, guaranteeing atomic execution (execute fully or not at all) and a Dockerfile generator to transfer successful configuration steps into runnable Dockerfiles. We evaluate Repo2Run~on our proposed benchmark of 420 recent Python repositories with unit tests, where it achieves an 86.0% success rate, outperforming the best baseline by 63.9%. Repo2Run is available at https://github.com/bytedance/Repo2Run.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01145v2">Dual Reasoning: A GNN-LLM Collaborative Framework for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at intuitive, implicit reasoning. Guiding LLMs to construct thought chains can enhance their deliberate reasoning abilities, but also faces challenges such as hallucination. Knowledge Graphs (KGs) can provide explicit structured knowledge for LLMs to alleviate these issues. However, existing KG-enhanced methods often overlook explicit graph learning, making it challenging to efficiently provide precise reasoning chains for LLMs. Following dual-process theory, we propose Dual-Reasoning (DualR), a novel framework that integrates an external system based on Graph Neural Network (GNN) for explicit reasoning on KGs, complementing the implicit reasoning of LLMs through externalized reasoning chains. DualR designs an LLM-empowered GNN module for explicit learning on KGs, efficiently extracting high-quality reasoning chains. These reasoning chains are then refined to a knowledge-enhanced multiple-choice prompt, guiding a frozen LLM to reason thoughtfully for final answer determination. Extensive experiments on three benchmark KGQA datasets demonstrate that DualR achieves state-of-the-art performance while maintaining high efficiency and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14074v2">Investigating Non-Transitivity in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 8 pages, 6 figures, 2 tables (30 pages, 11 figures, 8 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04130v1">Token-Efficient Long Video Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (\textbf{S}patiotemporal \textbf{TO}ken \textbf{R}eduction for \textbf{M}ultimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5\% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02398v2">Prompting with Phonemes: Enhancing LLMs' Multilinguality for Non-Latin Script Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted for NAACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Although multilingual LLMs have achieved remarkable performance across benchmarks, we find they continue to underperform on non-Latin script languages across contemporary LLM families. This discrepancy arises from the fact that LLMs are pretrained with orthographic scripts, which are dominated by Latin characters that obscure their shared phonology with non-Latin scripts. We propose leveraging phonemic transcriptions as complementary signals to induce script-invariant representations. Our study demonstrates that integrating phonemic signals improves performance across both non-Latin and Latin script languages, with a particularly significant impact on closing the performance gap between the two. Through detailed experiments, we show that phonemic and orthographic scripts retrieve distinct examples for in-context learning (ICL). This motivates our proposed Mixed-ICL retrieval strategy, where further aggregation from both leads to our significant performance improvements for both Latin script languages (up to 12.6%) and non-Latin script languages (up to 15.1%) compared to randomized ICL retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04113v1">Uncovering Gaps in How Humans and LLMs Interpret Subjective Language</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Published at ICLR 2025
    </div>
    <details class="paper-abstract">
      Humans often rely on subjective natural language to direct language models (LLMs); for example, users might instruct the LLM to write an enthusiastic blogpost, while developers might train models to be helpful and harmless using LLM-based edits. The LLM's operational semantics of such subjective phrases -- how it adjusts its behavior when each phrase is included in the prompt -- thus dictates how aligned it is with human intent. In this work, we uncover instances of misalignment between LLMs' actual operational semantics and what humans expect. Our method, TED (thesaurus error detector), first constructs a thesaurus that captures whether two phrases have similar operational semantics according to the LLM. It then elicits failures by unearthing disagreements between this thesaurus and a human-constructed reference. TED routinely produces surprising instances of misalignment; for example, Mistral 7B Instruct produces more harassing outputs when it edits text to be witty, and Llama 3 8B Instruct produces dishonest articles when instructed to make the articles enthusiastic. Our results demonstrate that humans can uncover unexpected LLM behavior by scrutinizing relationships between abstract concepts, without supervising outputs directly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18377v3">ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15692v2">DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has drawn attention to their potential for accelerating drug discovery. However, a central problem remains: translating theoretical ideas into robust implementations in the highly specialized context of pharmaceutical research. This limitation prevents practitioners from making full use of the latest AI developments in drug discovery. To address this challenge, we introduce DrugAgent, a multi-agent framework that automates machine learning (ML) programming for drug discovery tasks. DrugAgent employs an LLM Planner that formulates high-level ideas and an LLM Instructor that identifies and integrates domain knowledge when implementing those ideas. We present case studies on three representative drug discovery tasks. Our results show that DrugAgent consistently outperforms leading baselines, including a relative improvement of 4.92% in ROC-AUC compared to ReAct for drug-target interaction (DTI). DrugAgent is publicly available at https://anonymous.4open.science/r/drugagent-5C42/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03340v1">EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Theory-of-Mind (ToM), the ability to infer others' perceptions and mental states, is fundamental to human interaction but remains a challenging task for Large Language Models (LLMs). While existing ToM reasoning methods show promise with reasoning via perceptual perspective-taking, they often rely excessively on LLMs, reducing their efficiency and limiting their applicability to high-order ToM reasoning, which requires multi-hop reasoning about characters' beliefs. To address these issues, we present EnigmaToM, a novel neuro-symbolic framework that enhances ToM reasoning by integrating a Neural Knowledge Base of entity states (Enigma) for (1) a psychology-inspired iterative masking mechanism that facilitates accurate perspective-taking and (2) knowledge injection that elicits key entity information. Enigma generates structured representations of entity states, which construct spatial scene graphs -- leveraging spatial information as an inductive bias -- for belief tracking of various ToM orders and enhancing events with fine-grained entity state details. Experimental results on multiple benchmarks, including ToMi, HiToM, and FANToM, show that EnigmaToM significantly improves ToM reasoning across LLMs of varying sizes, particularly excelling in high-order reasoning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03313v1">LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05891v2">Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 The 40th ACM/SIGAPP Symposium On Applied Computing
    </div>
    <details class="paper-abstract">
      In education, the capability of generating human-like text of Large Language Models (LLMs) inspired work on how they can increase the efficiency of learning and teaching. We study the affordability of these models for educators and students by investigating how LLMs answer multiple-choice questions (MCQs) with respect to hardware constraints and refinement techniques. We explore this space by using generic pre-trained LLMs (the 7B, 13B, and 70B variants of LLaMA-2) to answer 162 undergraduate-level MCQs from a course on Programming Languages (PL) -- the MCQ dataset is a contribution of this work, which we make publicly available. Specifically, we dissect how different factors, such as using readily-available material -- (parts of) the course's textbook -- for fine-tuning and quantisation (to decrease resource usage) can change the accuracy of the responses. The main takeaway is that smaller textbook-based fine-tuned models outperform generic larger ones (whose pre-training requires conspicuous resources), making the usage of LLMs for answering MCQs resource- and material-wise affordable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09977v2">LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs -- No Silver Bullet for LC or RAG Routing</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Effectively incorporating external knowledge into Large Language Models (LLMs) is crucial for enhancing their capabilities and addressing real-world needs. Retrieval-Augmented Generation (RAG) offers an effective method for achieving this by retrieving the most relevant fragments into LLMs. However, the advancements in context window size for LLMs offer an alternative approach, raising the question of whether RAG remains necessary for effectively handling external knowledge. Several existing studies provide inconclusive comparisons between RAG and long-context (LC) LLMs, largely due to limitations in the benchmark designs. In this paper, we present LaRA, a novel benchmark specifically designed to rigorously compare RAG and LC LLMs. LaRA encompasses 2326 test cases across four practical QA task categories and three types of naturally occurring long texts. Through systematic evaluation of seven open-source and four proprietary LLMs, we find that the optimal choice between RAG and LC depends on a complex interplay of factors, including the model's parameter size, long-text capabilities, context length, task type, and the characteristics of the retrieved chunks. Our findings provide actionable guidelines for practitioners to effectively leverage both RAG and LC approaches in developing and deploying LLM applications. Our code and dataset is provided at: \href{https://github.com/Alibaba-NLP/LaRA}{\textbf{https://github.com/Alibaba-NLP/LaRA}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03261v1">Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can perform various natural language processing (NLP) tasks through in-context learning without relying on supervised data. However, multiple previous studies have reported suboptimal performance of LLMs in biological text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. To address these challenges, we experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our findings show that frontier LLMs can approach or surpass the performance of state-of-the-art (SOTA) BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these results, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09601v2">TimeRefine: Temporal Grounding with Time Refining Video LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Video temporal grounding aims to localize relevant temporal boundaries in a video given a textual prompt. Recent work has focused on enabling Video LLMs to perform video temporal grounding via next-token prediction of temporal timestamps. However, accurately localizing timestamps in videos remains challenging for Video LLMs when relying solely on temporal token prediction. Our proposed TimeRefine addresses this challenge in two ways. First, instead of directly predicting the start and end timestamps, we reformulate the temporal grounding task as a temporal refining task: the model first makes rough predictions and then refines them by predicting offsets to the target segment. This refining process is repeated multiple times, through which the model progressively self-improves its temporal localization accuracy. Second, to enhance the model's temporal perception capabilities, we incorporate an auxiliary prediction head that penalizes the model more if a predicted segment deviates further from the ground truth, thus encouraging the model to make closer and more accurate predictions. Our plug-and-play method can be integrated into most LLM-based temporal grounding approaches. The experimental results demonstrate that TimeRefine achieves 3.6% and 5.0% mIoU improvements on the ActivityNet and Charades-STA datasets, respectively. Code and pretrained models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01711v3">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 added project repository & dataset URL
    </div>
    <details class="paper-abstract">
      Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19622v2">Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 12 pages, 1 figure, 3 tables, 4 prompt/data templates
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have raised interest in their formal reasoning capabilities, particularly in mathematics. While closed LLMs like GPT-4 perform well on mathematical benchmarks, e.g., GSM8K, it remains unclear whether small to medium-sized open LLMs can achieve similar performance, questioning their reliability. To close this gap, we propose a post-training approach leveraging a mixture of opinions (MoO) from weaker ancillary LLMs to enhance a (relatively) stronger LLM's reasoning. For that, each post-training sample is augmented with Chain-of-Thought (CoT) reasoning steps and answers from ancillary LLMs, enabling the main LLM to learn from diverse perspectives. We compare MoO with standard supervised fine-tuning (SFT), few-shot prompting, and the Mixture of Agents (MoA) method on mathematical reasoning benchmarks. Our results show that incorporating weaker LLMs' opinions improves mathematical reasoning by an average of 5%, highlighting the value of diverse perspectives in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14644v2">zsLLMCode: An Effective Approach for Code Embedding via LLM with Zero-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has greatly advanced artificial intelligence (AI) in software engineering (SE), with code embeddings playing a critical role in tasks like code-clone detection and code clustering. However, existing methods for code embedding, including those based on LLMs, often depend on costly supervised training or fine-tuning for domain adaptation. This paper proposes a novel zero-shot approach, zsLLMCode, to generate code embeddings by using LLMs and sentence embedding models. This approach attempts to eliminate the need for task-specific training or fine-tuning, and to effectively address the issue of erroneous information commonly found in LLM-generated outputs. We conducted a series of experiments to evaluate the performance of the proposed approach by considering various LLMs and embedding models. The results have demonstrated the effectiveness and superiority of our method zsLLMCode over state-of-the-art unsupervised approaches such as SourcererCC, Code2vec, InferCode, and TransformCode. Our findings highlight the potential of zsLLMCode to advance the field of SE by providing robust and efficient solutions for code embedding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03194v1">Structured Outputs Enable General-Purpose LLMs to be Medical Experts</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Medical question-answering (QA) is a critical task for evaluating how effectively large language models (LLMs) encode clinical knowledge and assessing their potential applications in medicine. Despite showing promise on multiple-choice tests, LLMs frequently struggle with open-ended medical questions, producing responses with dangerous hallucinations or lacking comprehensive coverage of critical aspects. Existing approaches attempt to address these challenges through domain-specific fine-tuning, but this proves resource-intensive and difficult to scale across models. To improve the comprehensiveness and factuality of medical responses, we propose a novel approach utilizing structured medical reasoning. Our method guides LLMs through an seven-step cognitive process inspired by clinical diagnosis, enabling more accurate and complete answers without additional training. Experiments on the MedLFQA benchmark demonstrate that our approach achieves the highest Factuality Score of 85.8, surpassing fine-tuned models. Notably, this improvement transfers to smaller models, highlighting the method's efficiency and scalability. Our code and datasets are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03180v1">Enhancing Cybersecurity in Critical Infrastructure with LLM-Assisted Explainable IoT Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Ensuring the security of critical infrastructure has become increasingly vital with the proliferation of Internet of Things (IoT) systems. However, the heterogeneous nature of IoT data and the lack of human-comprehensible insights from anomaly detection models remain significant challenges. This paper presents a hybrid framework that combines numerical anomaly detection using Autoencoders with Large Language Models (LLMs) for enhanced preprocessing and interpretability. Two preprocessing approaches are implemented: a traditional method utilizing Principal Component Analysis (PCA) to reduce dimensionality and an LLM-assisted method where GPT-4 dynamically recommends feature selection, transformation, and encoding strategies. Experimental results on the KDDCup99 10% corrected dataset demonstrate that the LLM-assisted preprocessing pipeline significantly improves anomaly detection performance. The macro-average F1 score increased from 0.49 in the traditional PCA-based approach to 0.98 with LLM-driven insights. Additionally, the LLM generates natural language explanations for detected anomalies, providing contextual insights into their causes and implications. This framework highlights the synergy between numerical AI models and LLMs, delivering an accurate, interpretable, and efficient solution for IoT cybersecurity in critical infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07810v5">Transformer Block Coupling and its Correlation with Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Published as a conference paper at the International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language processing, and a precise understanding of the internal mechanisms driving their success is essential. In this work, we analyze the trajectories of token embeddings as they pass through transformer blocks, linearizing the system along these trajectories through their Jacobian matrices. By examining the relationships between these block Jacobians, we uncover the phenomenon of \textbf{transformer block coupling} in a multitude of LLMs, characterized by the coupling of their top singular vectors across tokens and depth. Our findings reveal that coupling \textit{positively correlates} with model performance, and that this relationship is stronger than with other hyperparameters such as parameter count, model depth, and embedding dimension. We further investigate how these properties emerge during training, observing a progressive development of coupling, increased linearity, and layer-wise exponential growth in token trajectories. Additionally, experiments with Vision Transformers (ViTs) corroborate the emergence of coupling and its relationship with generalization, reinforcing our findings in LLMs. Collectively, these insights offer a novel perspective on token interactions in transformers, opening new directions for studying their mechanisms as well as improving training and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v3">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v2">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v4">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v1">SoK: Knowledge is All You Need: Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03074v1">BEVDriver: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Autonomous driving has the potential to set the stage for more efficient future mobility, requiring the research domain to establish trust through safe, reliable and transparent driving. Large Language Models (LLMs) possess reasoning capabilities and natural language understanding, presenting the potential to serve as generalized decision-makers for ego-motion planning that can interact with humans and navigate environments designed for human drivers. While this research avenue is promising, current autonomous driving approaches are challenged by combining 3D spatial grounding and the reasoning and language capabilities of LLMs. We introduce BEVDriver, an LLM-based model for end-to-end closed-loop driving in CARLA that utilizes latent BEV features as perception input. BEVDriver includes a BEV encoder to efficiently process multi-view images and 3D LiDAR point clouds. Within a common latent space, the BEV features are propagated through a Q-Former to align with natural language instructions and passed to the LLM that predicts and plans precise future trajectories while considering navigation instructions and critical scenarios. On the LangAuto benchmark, our model reaches up to 18.9% higher performance on the Driving Score compared to SoTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10877v2">Improving Data Efficiency via Curating LLM-Driven Rating Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Instruction tuning is critical for adapting large language models (LLMs) to downstream tasks, and recent studies have demonstrated that small amounts of human-curated data can outperform larger datasets, challenging traditional data scaling laws. While LLM-based data quality rating systems offer a cost-effective alternative to human annotation, they often suffer from inaccuracies and biases, even in powerful models like GPT-4. In this work, we introduce DS2, a Diversity-aware Score curation method for Data Selection. By systematically modeling error patterns through a score transition matrix, DS2 corrects LLM-based scores and promotes diversity in the selected data samples. Our approach shows that a curated subset (just 3.3% of the original dataset) outperforms full-scale datasets (300k samples) across various machine-alignment benchmarks, and matches or surpasses human-aligned datasets such as LIMA with the same sample size (1k samples). These findings challenge conventional data scaling assumptions, highlighting that redundant, low-quality samples can degrade performance and reaffirming that "more can be less."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03969v1">Trim My View: An LLM-Based Code Query System for Module Retrieval in Robotic Firmware</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The software compilation process has a tendency to obscure the original design of the system and makes it difficult both to identify individual components and discern their purpose simply by examining the resulting binary code. Although decompilation techniques attempt to recover higher-level source code from the machine code in question, they are not fully able to restore the semantics of the original functions. Furthermore, binaries are often stripped of metadata, and this makes it challenging to reverse engineer complex binary software. In this paper we show how a combination of binary decomposition techniques, decompilation passes, and LLM-powered function summarization can be used to build an economical engine to identify modules in stripped binaries and associate them with high-level natural language descriptions. We instantiated this technique with three underlying open-source LLMs -- CodeQwen, DeepSeek-Coder and CodeStral -- and measured its effectiveness in identifying modules in robotics firmware. This experimental evaluation involved 467 modules from four devices from the ArduPilot software suite, and showed that CodeStral, the best-performing backend LLM, achieves an average F1-score of 0.68 with an online running time of just a handful of seconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03967v1">Model Behavior Specification by Leveraging LLM Self-Playing and Self-Improving</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Training AI models is challenging, particularly when crafting behavior instructions. Traditional methods rely on machines (supervised learning) or manual pattern discovery, which results in not interpretable models or time sink. While Large Language Models (LLMs) simplify instruction writing through natural language, articulating intended model behavior still remains difficult. We introduce Visionary Tuning, a human-in-the-loop self-playing followed by automatic self-refinement to improve behavior specification. Our system helps users clarify desired behavior through self-playing and generates prompts through self-improving, Our first evaluation involves user study conducted on a system implementation of Visionary Tuning within the context of chatbot behavior. Our system self-play itself by simulating user interactions to identify patterns and create effective prompts based on the pattern. In a within-subject study (N=12), participants pinpointed more patterns through self-playing and crafted better prompts. Surprisingly, users felt more or less success level in specifying the model behavior. Follow-up crowd studies (N=60) confirmed that the chatbot adhered to instructions without sacrificing quality. Our second evaluation is a case study on a real-world implementation using a movie rating dataset with Visionary Tuning, demonstrating its effectiveness and robustness in modeling a critic's preferences across the spectrum of low to highly rated movies. Together, these results suggest how AI improves the design process of interactive AI systems. Furthermore, they suggest how the benefits of these tools may be non-obvious to end-users. We reflect on these findings and suggest future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14973v4">GenCeption: Evaluate Vision LLMs with Unlabeled Unimodal Data</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Published by Computer Speech & Language (https://doi.org/10.1016/j.csl.2025.101785). Source code and Leaderboard: https://github.com/llcresearch/GenCeption
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) are typically assessed using expensive annotated multimodal benchmarks, which often lag behind the rapidly evolving demands of MLLM evaluation. This paper outlines and validates GenCeption, a novel, annotation-free evaluation method that requires only unimodal data to measure inter-modality semantic coherence and inversely assesses MLLMs' tendency to hallucinate. This approach eliminates the need for costly data annotation, minimizes the risk of training data contamination, is expected to result in slower benchmark saturation, and avoids the illusion of emerging abilities. Inspired by the DrawCeption game, GenCeption begins with a non-textual sample and proceeds through iterative description and generation steps. The semantic drift across iterations is quantified using the GC@T metric. While GenCeption is principally applicable to MLLMs across various modalities, this paper focuses on its implementation and validation for Vision LLMs (VLLMs). Based on the GenCeption method, we establish the MMECeption benchmark for evaluating VLLMs, and compare the performance of several popular VLLMs and human annotators. Our empirical results validate GenCeption's effectiveness, demonstrating strong correlations with established VLLM benchmarks. VLLMs still significantly lag behind human performance and struggle especially with text-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03911v1">Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in robotic systems presents unique safety challenges, particularly in unpredictable environments. Although LLMs, leveraging zero-shot learning, enhance human-robot interaction and decision-making capabilities, their inherent probabilistic nature and lack of formal guarantees raise significant concerns for safety-critical applications. Traditional model-based verification approaches often rely on precise system models, which are difficult to obtain for real-world robotic systems and may not be fully trusted due to modeling inaccuracies, unmodeled dynamics, or environmental uncertainties. To address these challenges, this paper introduces a safety assurance framework for LLM-controlled robots based on data-driven reachability analysis, a formal verification technique that ensures all possible system trajectories remain within safe operational limits. Our framework specifically investigates the problem of instructing an LLM to navigate the robot to a specified goal and assesses its ability to generate low-level control actions that successfully guide the robot safely toward that goal. By leveraging historical data to construct reachable sets of states for the robot-LLM system, our approach provides rigorous safety guarantees against unsafe behaviors without relying on explicit analytical models. We validate the framework through experimental case studies in autonomous navigation and task planning, demonstrating its effectiveness in mitigating risks associated with LLM-generated commands. This work advances the integration of formal methods into LLM-based robotics, offering a principled and practical approach to ensuring safety in next-generation autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03889v1">Pretrained LLMs as Real-Time Controllers for Robot Operated Serial Production Line</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The manufacturing industry is undergoing a transformative shift, driven by cutting-edge technologies like 5G, AI, and cloud computing. Despite these advancements, effective system control, which is crucial for optimizing production efficiency, remains a complex challenge due to the intricate, knowledge-dependent nature of manufacturing processes and the reliance on domain-specific expertise. Conventional control methods often demand heavy customization, considerable computational resources, and lack transparency in decision-making. In this work, we investigate the feasibility of using Large Language Models (LLMs), particularly GPT-4, as a straightforward, adaptable solution for controlling manufacturing systems, specifically, mobile robot scheduling. We introduce an LLM-based control framework to assign mobile robots to different machines in robot assisted serial production lines, evaluating its performance in terms of system throughput. Our proposed framework outperforms traditional scheduling approaches such as First-Come-First-Served (FCFS), Shortest Processing Time (SPT), and Longest Processing Time (LPT). While it achieves performance that is on par with state-of-the-art methods like Multi-Agent Reinforcement Learning (MARL), it offers a distinct advantage by delivering comparable throughput without the need for extensive retraining. These results suggest that the proposed LLM-based solution is well-suited for scenarios where technical expertise, computational resources, and financial investment are limited, while decision transparency and system scalability are critical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v2">A generative approach to LLM harmfulness detection with special red flag tokens</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09345v3">Rational Tuning of LLM Cascades via Probabilistic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using grid search, our parametric Markov-copula model significantly improves runtime scaling with respect to the length of the cascade and the desired resolution of the cost-error curve, turning them from intractable into low-order polynomial. In addition, the optimal thresholds computed using our continuous optimization-based algorithm increasingly outperform those found via grid search as cascade length grows, improving the area under the cost-error curve by 1.9% on average for cascades consisting of at least three models. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04840v1">Framing the Game: How Context Shapes LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04830v1">Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04828v1">Beyond Next Word Prediction: Developing Comprehensive Evaluation Frameworks for measuring LLM performance on real world applications</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are fundamentally next-token prediction systems, their practical applications extend far beyond this basic function. From natural language processing and text generation to conversational assistants and software use, LLMs have numerous use-cases, and have already acquired a significant degree of enterprise adoption. To evaluate such models, static evaluation datasets, consisting of a set of prompts and their corresponding ground truths, are often used to benchmark the efficacy of the model for a particular task. In this paper, we provide the basis for a more comprehensive evaluation framework, based upon a traditional game and tool-based architecture that enables a more overarching measurement of a model's capabilities. For simplicity, we provide a generalized foundation that can be extended, without significant alteration, to numerous scenarios, from specific use cases such as supply chain management or financial reasoning, to abstract measurements such as ethics or safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01048v3">Personalize Your LLM: Fake it then Align it</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is essential for delivering tailored interactions that improve user experience. Many existing personalization methods require fine-tuning LLMs for each user, rendering them prohibitively expensive for widespread adoption. Although retrieval-based approaches offer a more compute-efficient alternative, they still depend on large, high-quality datasets that are not consistently available for all users. To address this challenge, we propose CHAMELEON, a scalable and efficient personalization approach that uses (1) self-generated personal preference data and (2) representation editing to enable quick and cost-effective personalization. Our experiments on various tasks, including those from the LaMP personalization benchmark, show that CHAMELEON efficiently adapts models to personal preferences, improving instruction-tuned models and outperforms two personalization baselines by an average of 40% across two model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07132v2">Interactive Data Harmonization with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Data harmonization is an essential task that entails integrating datasets from diverse sources. Despite years of research in this area, it remains a time-consuming and challenging task due to schema mismatches, varying terminologies, and differences in data collection methodologies. This paper presents the case for agentic data harmonization as a means to both empower experts to harmonize their data and to streamline the process. We introduce Harmonia, a system that combines LLM-based reasoning, an interactive user interface, and a library of data harmonization primitives to automate the synthesis of data harmonization pipelines. We demonstrate Harmonia in a clinical data harmonization scenario, where it helps to interactively create reusable pipelines that map datasets to a standard format. Finally, we discuss challenges and open problems, and suggest research directions for advancing our vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11620v2">Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 18 pages and 3 References Pages
    </div>
    <details class="paper-abstract">
      In this work, we explore uncertainty estimation as a proxy for correctness in LLM-generated code. To this end, we adapt two state-of-the-art techniques from natural language generation -- one based on entropy and another on mutual information -- to the domain of code generation. Given the distinct semantic properties of code, we introduce modifications, including a semantic equivalence check based on symbolic execution. Our findings indicate a strong correlation between the uncertainty computed through these techniques and correctness, highlighting the potential of uncertainty estimation for quality assessment. Additionally, we propose a simplified version of the entropy-based method that assumes a uniform distribution over the LLM's responses, demonstrating comparable effectiveness. Using these techniques, we develop an abstention policy that prevents the model from making predictions when uncertainty is high, reducing incorrect outputs to near zero. Our evaluation on the LiveCodeBench shows that our approach significantly outperforms a baseline relying solely on LLM-reported log-probabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03710v1">Improving LLM Safety Alignment with Dual-Objective Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03705v1">Effective LLM Knowledge Learning via Model Generalization</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on enormous documents that contain extensive world knowledge. However, it is still not well-understood how knowledge is acquired via autoregressive pre-training. This lack of understanding greatly hinders effective knowledge learning, especially for continued pretraining on up-to-date information, as this evolving information often lacks diverse repetitions like foundational knowledge. In this paper, we focus on understanding and improving LLM knowledge learning. We found and verified that knowledge learning for LLMs can be deemed as an implicit supervised task hidden in the autoregressive pre-training objective. Our findings suggest that knowledge learning for LLMs would benefit from methods designed to improve generalization ability for supervised tasks. Based on our analysis, we propose the formatting-based data augmentation to grow in-distribution samples, which does not present the risk of altering the facts embedded in documents as text paraphrasing. We also introduce sharpness-aware minimization as an effective optimization algorithm to better improve generalization. Moreover, our analysis and method can be readily extended to instruction tuning. Extensive experiment results validate our findings and demonstrate our methods' effectiveness in both continued pre-training and instruction tuning. This paper offers new perspectives and insights to interpret and design effective strategies for LLM knowledge learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03704v1">A Practical Memory Injection Attack against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03686v1">MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 26 pages, 7 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have shown significant potential in tackling diverse tasks. However, to design effective MAS, existing approaches heavily rely on manual configurations or multiple calls of advanced LLMs, resulting in inadaptability and high inference costs. In this paper, we simplify the process of building an MAS by reframing it as a generative language task, where the input is a user query and the output is a corresponding MAS. To address this novel task, we unify the representation of MAS as executable code and propose a consistency-oriented data construction pipeline to create a high-quality dataset comprising coherent and consistent query-MAS pairs. Using this dataset, we train MAS-GPT, an open-source medium-sized LLM that is capable of generating query-adaptive MAS within a single LLM inference. The generated MAS can be seamlessly applied to process user queries and deliver high-quality responses. Extensive experiments on 9 benchmarks and 5 LLMs show that the proposed MAS-GPT consistently outperforms 10+ baseline MAS methods on diverse settings, indicating MAS-GPT's high effectiveness, efficiency and strong generalization ability. Code will be available at https://github.com/rui-ye/MAS-GPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03594v1">Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      While LLMs have demonstrated remarkable potential in time series forecasting, their practical deployment remains constrained by excessive computational demands and memory footprints. Existing LLM-based approaches typically suffer from three critical limitations: Inefficient parameter utilization in handling numerical time series patterns; Modality misalignment between continuous temporal signals and discrete text embeddings; and Inflexibility for real-time expert knowledge integration. We present SMETimes, the first systematic investigation of sub-3B parameter SLMs for efficient and accurate time series forecasting. Our approach centers on three key innovations: A statistically-enhanced prompting mechanism that bridges numerical time series with textual semantics through descriptive statistical features; A adaptive fusion embedding architecture that aligns temporal patterns with language model token spaces through learnable parameters; And a dynamic mixture-of-experts framework enabled by SLMs' computational efficiency, adaptively combining base predictions with domain-specific models. Extensive evaluations across seven benchmark datasets demonstrate that our 3B-parameter SLM achieves state-of-the-art performance on five primary datasets while maintaining 3.8x faster training and 5.2x lower memory consumption compared to 7B-parameter LLM baselines. Notably, the proposed model exhibits better learning capabilities, achieving 12.3% lower MSE than conventional LLM. Ablation studies validate that our statistical prompting and cross-modal fusion modules respectively contribute 15.7% and 18.2% error reduction in long-horizon forecasting tasks. By redefining the efficiency-accuracy trade-off landscape, this work establishes SLMs as viable alternatives to resource-intensive LLMs for practical time series forecasting. Code and models are available at https://github.com/xiyan1234567/SMETimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03592v1">English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      For consumer usage of locally deployed LLMs, the GGUF format and k_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to k_quantization yielded non-significant results (In all cases p > 0.237) indicating that current quantization practices do not disproportionately harm multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16207v3">From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO and have made significant progress. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and breaks it down into sub-tasks. We constructed 18k high-quality instruction-response pairs across five formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) by distilling gpt-4o and evaluated against ten open-sourced LLMs, including recent popular DeepSeek-R1. We also fine-tuned several 7~8B small models to achieve comparable performance with Deepseek-R1-671B. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding capabilities. Fine-tuned models are released at https: //huggingface.co/fm-universe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03587v1">"You don't need a university degree to comprehend data protection this way": LLM-Powered Interactive Privacy Policy Assessment</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 6 pages without appendices and references, 12 pages total, 3 figures, poster at CHI 2025. arXiv admin note: text overlap with arXiv:2501.16033
    </div>
    <details class="paper-abstract">
      Protecting online privacy requires users to engage with and comprehend website privacy policies, but many policies are difficult and tedious to read. We present the first qualitative user study on Large Language Model (LLM)-driven privacy policy assessment. To this end, we build and evaluate an LLM-based privacy policy assessment browser extension, which helps users understand the essence of a lengthy, complex privacy policy while browsing. The tool integrates a dashboard and an LLM chat. In our qualitative user study (N=22), we evaluate usability, understandability of the information our tool provides, and its impacts on awareness. While providing a comprehensible quick overview and a chat for in-depth discussion improves privacy awareness, users note issues with building trust in the tool. From our insights, we derive important design implications to guide future policy analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03586v1">Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in software vulnerability detection, particularly on function-level benchmarks like Devign and BigVul. However, real-world detection requires interprocedural analysis, as vulnerabilities often emerge through multi-hop function calls rather than isolated functions. While repository-level benchmarks like ReposVul and VulEval introduce interprocedural context, they remain computationally expensive, lack pairwise evaluation of vulnerability fixes, and explore limited context retrieval, limiting their practicality. We introduce JitVul, a JIT vulnerability detection benchmark linking each function to its vulnerability-introducing and fixing commits. Built from 879 CVEs spanning 91 vulnerability types, JitVul enables comprehensive evaluation of detection capabilities. Our results show that ReAct Agents, leveraging thought-action-observation and interprocedural context, perform better than LLMs in distinguishing vulnerable from benign code. While prompting strategies like Chain-of-Thought help LLMs, ReAct Agents require further refinement. Both methods show inconsistencies, either misidentifying vulnerabilities or over-analyzing security guards, indicating significant room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16205v5">LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07115v3">Online Scheduling for LLM Inference with KV Cache Constraints</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Will add a lemma in the proof of Theorem 5.3 to make the statement and proof more rigorous
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference, where a trained model generates text one word at a time in response to user prompts, is a computationally intensive process requiring efficient scheduling to optimize latency and resource utilization. A key challenge in LLM inference is the management of the Key-Value (KV) cache, which reduces redundant computations but introduces memory constraints. In this work, we model LLM inference with KV cache constraints theoretically and propose novel batching and scheduling algorithms that minimize inference latency while effectively managing the KV cache's memory. We analyze both semi-online and fully online scheduling models, and our results are threefold. First, we provide a polynomial-time algorithm that achieves exact optimality in terms of average latency in the semi-online prompt arrival model. Second, in the fully online case with a stochastic prompt arrival, we introduce an efficient online scheduling algorithm with constant regret. Third, we prove that no algorithm (deterministic or randomized) can achieve a constant competitive ratio in fully online adversarial settings. Our empirical evaluations on a public LLM inference dataset, using the Llama-70B model on A100 GPUs, show that our approach significantly outperforms benchmark algorithms used currently in practice, achieving lower latency while reducing energy consumption. Overall, our results offer a path toward more sustainable and cost-effective LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03505v1">Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Model(LLM)-based Multi-Agent Systems(MAS) have demonstrated remarkable potential for tackling complex decision-making tasks. However, existing frameworks inevitably rely on serialized execution paradigms, where agents must complete sequential LLM planning before taking action. This fundamental constraint severely limits real-time responsiveness and adaptation, which is crucial in dynamic environments with ever-changing scenarios. In this paper, we propose a novel parallelized planning-acting framework for LLM-based MAS, featuring a dual-thread architecture with interruptible execution to enable concurrent planning and acting. Specifically, our framework comprises two core threads:(1) a planning thread driven by a centralized memory system, maintaining synchronization of environmental states and agent communication to support dynamic decision-making; and (2) an acting thread equipped with a comprehensive skill library, enabling automated task execution through recursive decomposition. Extensive experiments on challenging Minecraft demonstrate the effectiveness of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03503v1">Collaborative Expert LLMs Guided Multi-Objective Molecular Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Molecular optimization is a crucial yet complex and time-intensive process that often acts as a bottleneck for drug development. Traditional methods rely heavily on trial and error, making multi-objective optimization both time-consuming and resource-intensive. Current AI-based methods have shown limited success in handling multi-objective optimization tasks, hampering their practical utilization. To address this challenge, we present MultiMol, a collaborative large language model (LLM) system designed to guide multi-objective molecular optimization. MultiMol comprises two agents, including a data-driven worker agent and a literature-guided research agent. The data-driven worker agent is a large language model being fine-tuned to learn how to generate optimized molecules considering multiple objectives, while the literature-guided research agent is responsible for searching task-related literature to find useful prior knowledge that facilitates identifying the most promising optimized candidates. In evaluations across six multi-objective optimization tasks, MultiMol significantly outperforms existing methods, achieving a 82.30% success rate, in sharp contrast to the 27.50% success rate of current strongest methods. To further validate its practical impact, we tested MultiMol on two real-world challenges. First, we enhanced the selectivity of Xanthine Amine Congener (XAC), a promiscuous ligand that binds both A1R and A2AR, successfully biasing it towards A1R. Second, we improved the bioavailability of Saquinavir, an HIV-1 protease inhibitor with known bioavailability limitations. Overall, these results indicate that MultiMol represents a highly promising approach for multi-objective molecular optimization, holding great potential to accelerate the drug development process and contribute to the advancement of pharmaceutical research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03459v1">Unified Mind Model: Reimagining Autonomous Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive abilities.Such human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of LLMs.Although there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10069v2">A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      LLM test-time compute (or LLM inference) via search has emerged as a promising research area with rapid developments. However, current frameworks often adopt distinct perspectives on three key aspects (task definition, LLM profiling, and search procedures), making direct comparisons challenging. Moreover, the search algorithms employed often diverge from standard implementations, and their specific characteristics are not thoroughly specified. In this survey, we provide a comprehensive technical review that unifies task definitions and provides modular definitions of LLM profiling and search procedures. The definitions enable precise comparisons of various LLM inference frameworks while highlighting their departures from conventional search algorithms. We also discuss the applicability, performance, and efficiency of these methods. We have updated our content to include the latest papers, and the differences between versions are highlighted in the appendix. For further details and ongoing updates, please refer to our GitHub repository: https://github.com/xinzhel/LLM-Agent-Survey/blob/main/search.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03064v1">Improving LLM-as-a-Judge Inference with the Judgment Distribution</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Using language models to scalably approximate human preferences on text quality (LLM-as-a-judge) has become a standard practice applicable to many tasks. A judgment is often extracted from the judge's textual output alone, typically with greedy decoding. However, LLM judges naturally provide distributions over judgment tokens, inviting a breadth of inference methods for extracting fine-grained preferences. We find that taking the mean of the judgment distribution consistently outperforms taking the mode (i.e. greedy decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise). We further explore novel methods of deriving preferences from judgment distributions, and find that methods incorporating risk aversion often improve performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought (CoT) prompting, showing that CoT can collapse the spread of the judgment distribution, often harming performance. Our findings suggest leveraging distributional output can improve LLM-as-a-judge, as opposed to using the text interface alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v3">Internal Activation as the Polar Star for Steering Unsafe LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities across a wide range of tasks but also pose significant risks due to their potential to generate harmful content. Although existing safety mechanisms can improve model safety, they often lead to overly cautious behavior and fail to fully utilize LLMs' internal cognitive processes. Drawing inspiration from cognitive science, where humans rely on reflective reasoning (System 2 thinking) to regulate language and behavior, we empirically demonstrate that LLMs also possess a similar capacity for internal assessment and regulation, which can be actively detected. Building on this insight, we introduce SafeSwitch, a framework that dynamically regulates unsafe outputs by monitoring and utilizing the model's internal states. Our empirical results show that SafeSwitch reduces harmful outputs by over 80% on safety benchmarks while maintaining strong utility. Compared to traditional safety alignment methods, SafeSwitch delivers more informative and context-aware refusals, demonstrates resilience to unseen queries, and achieves these benefits while only tuning less than 6% of the original parameters. These features make SafeSwitch a promising approach for implementing nuanced safety controls in LLMs. Codes for this work are available at https://github.com/Hanpx20/SafeSwitch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03039v1">LLM Misalignment via Adversarial RLHF Platforms</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03032v1">SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Despite the state-of-the-art performance of Large Language Models (LLMs), these models often suffer from hallucinations, which can undermine their performance in critical applications. In this work, we propose SAFE, a novel method for detecting and mitigating hallucinations by leveraging Sparse Autoencoders (SAEs). While hallucination detection techniques and SAEs have been explored independently, their synergistic application in a comprehensive system, particularly for hallucination-aware query enrichment, has not been fully investigated. To validate the effectiveness of SAFE, we evaluate it on two models with available SAEs across three diverse cross-domain datasets designed to assess hallucination problems. Empirical results demonstrate that SAFE consistently improves query generation accuracy and mitigates hallucinations across all datasets, achieving accuracy improvements of up to 29.45%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11856v3">Automatically Improving LLM-based Verilog Generation using EDA Tool Feedback</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted for publication in TODAES Special Issue on Large Language Models for Electronic System Design Automation
    </div>
    <details class="paper-abstract">
      Traditionally, digital hardware designs are written in the Verilog hardware description language (HDL) and debugged manually by engineers. This can be time-consuming and error-prone for complex designs. Large Language Models (LLMs) are emerging as a potential tool to help generate fully functioning HDL code, but most works have focused on generation in the single-shot capacity: i.e., run and evaluate, a process that does not leverage debugging and, as such, does not adequately reflect a realistic development process. In this work, we evaluate the ability of LLMs to leverage feedback from electronic design automation (EDA) tools to fix mistakes in their own generated Verilog. To accomplish this, we present an open-source, highly customizable framework, AutoChip, which combines conversational LLMs with the output from Verilog compilers and simulations to iteratively generate and repair Verilog. To determine the success of these LLMs we leverage the VerilogEval benchmark set. We evaluate four state-of-the-art conversational LLMs, focusing on readily accessible commercial models. EDA tool feedback proved to be consistently more effective than zero-shot prompting only with GPT-4o, the most computationally complex model we evaluated. In the best case, we observed a 5.8% increase in the number of successful designs with a 34.2% decrease in cost over the best zero-shot results. Mixing smaller models with this larger model at the end of the feedback iterations resulted in equally as much success as with GPT-4o using feedback, but incurred 41.9% lower cost (corresponding to an overall decrease in cost over zero-shot by 89.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02989v1">Effectively Steer LLM To Follow Preference via Building Confident Directions</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Having an LLM that aligns with human preferences is essential for accommodating individual needs, such as maintaining writing style or generating specific topics of interest. The majority of current alignment methods rely on fine-tuning or prompting, which can be either costly or difficult to control. Model steering algorithms, which modify the model output by constructing specific steering directions, are typically easy to implement and optimization-free. However, their capabilities are typically limited to steering the model into one of the two directions (i.e., bidirectional steering), and there has been no theoretical understanding to guarantee their performance. In this work, we propose a theoretical framework to understand and quantify the model steering methods. Inspired by the framework, we propose a confident direction steering method (CONFST) that steers LLMs via modifying their activations at inference time. More specifically, CONFST builds a confident direction that is closely aligned with users' preferences, and this direction is then added to the activations of the LLMs to effectively steer the model output. Our approach offers three key advantages over popular bidirectional model steering methods: 1) It is more powerful, since multiple (i.e. more than two) users' preferences can be aligned simultaneously; 2) It is simple to implement, since there is no need to determine which layer to add the steering vector to; 3) No explicit user instruction is required. We validate our method on GPT-2 XL (1.5B), Mistral (7B) and Gemma-it (9B) models for tasks that require shifting the output of LLMs across various topics and styles, achieving superior performance over competing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18282v2">Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 under review
    </div>
    <details class="paper-abstract">
      The increased adoption of Large Language Models (LLMs) and their potential to shape public opinion have sparked interest in assessing these models' political leanings. Building on previous research that compared LLMs and human opinions and observed political bias in system responses, we take a step further to investigate the underlying causes of such biases by empirically examining how the values and biases embedded in training corpora shape model outputs. Specifically, we propose a method to quantitatively evaluate political leanings embedded in the large pretraining corpora. Subsequently we investigate to whom are the LLMs' political leanings more aligned with, their pretrainig corpora or the surveyed human opinions. As a case study, we focus on probing the political leanings of LLMs in 32 U.S. Supreme Court cases, addressing contentious topics such as abortion and voting rights. Our findings reveal that LLMs strongly reflect the political leanings in their training data, and no strong correlation is observed with their alignment to human opinions as expressed in surveys. These results underscore the importance of responsible curation of training data and the need for robust evaluation metrics to ensure LLMs' alignment with human-centered values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03777v1">FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 9 pages, 5 figures, to be published in EuroMLSys '25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face challenges for on-device inference due to high memory demands. Traditional methods to reduce memory usage often compromise performance and lack adaptability. We propose FlexInfer, an optimized offloading framework for on-device inference, addressing these issues with techniques like asynchronous prefetching, balanced memory locking, and flexible tensor preservation. These strategies enhance memory efficiency and mitigate I/O bottlenecks, ensuring high performance within user-specified resource constraints. Experiments demonstrate that FlexInfer significantly improves throughput under limited resources, achieving up to 12.5 times better performance than existing methods and facilitating the deployment of large models on resource-constrained devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02879v1">Wikipedia in the Era of LLMs: Evolution and Risks</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 We release all the experimental dataset and source code at: https://github.com/HSM316/LLM_Wikipedia
    </div>
    <details class="paper-abstract">
      In this paper, we present a thorough analysis of the impact of Large Language Models (LLMs) on Wikipedia, examining the evolution of Wikipedia through existing data and using simulations to explore potential risks. We begin by analyzing page views and article content to study Wikipedia's recent changes and assess the impact of LLMs. Subsequently, we evaluate how LLMs affect various Natural Language Processing (NLP) tasks related to Wikipedia, including machine translation and retrieval-augmented generation (RAG). Our findings and simulation results reveal that Wikipedia articles have been influenced by LLMs, with an impact of approximately 1%-2% in certain categories. If the machine translation benchmark based on Wikipedia is influenced by LLMs, the scores of the models may become inflated, and the comparative results among models might shift as well. Moreover, the effectiveness of RAG might decrease if the knowledge base becomes polluted by LLM-generated content. While LLMs have not yet fully changed Wikipedia's language and knowledge structures, we believe that our empirical findings signal the need for careful consideration of potential future risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02863v1">Calibrating LLM Confidence with Semantic Steering: A Multi-Prompt Aggregation Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit misaligned confidence scores, usually overestimating the reliability of their predictions. While verbalized confidence in Large Language Models (LLMs) has gained attention, prior work remains divided on whether confidence scores can be systematically steered through prompting. Recent studies even argue that such prompt-induced confidence shifts are negligible, suggesting LLMs' confidence calibration is rigid to linguistic interventions. Contrary to these claims, we first rigorously confirm the existence of directional confidence shifts by probing three models (including GPT3.5, LLAMA3-70b, GPT4) across 7 benchmarks, demonstrating that explicit instructions can inflate or deflate confidence scores in a regulated manner. Based on this observation, we propose a novel framework containing three components: confidence steering, steered confidence aggregation and steered answers selection, named SteeringConf. Our method, SteeringConf, leverages a confidence manipulation mechanism to steer the confidence scores of LLMs in several desired directions, followed by a summarization module that aggregates the steered confidence scores to produce a final prediction. We evaluate our method on 7 benchmarks and it consistently outperforms the baselines in terms of calibration metrics in task of confidence calibration and failure detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02851v1">Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to hallucinate, a phenomenon often linked to creativity. While previous research has primarily explored this connection through theoretical or qualitative lenses, our work takes a quantitative approach to systematically examine the relationship between hallucination and creativity in LLMs. Given the complex nature of creativity, we propose a narrow definition tailored to LLMs and introduce an evaluation framework, HCL, which quantifies Hallucination and Creativity across different Layers of LLMs during decoding. Our empirical analysis reveals a tradeoff between hallucination and creativity that is consistent across layer depth, model type, and model size. Notably, across different model architectures, we identify a specific layer at each model size that optimally balances this tradeoff. Additionally, the optimal layer tends to appear in the early layers of larger models, and the confidence of the model is also significantly higher at this layer. These findings provide a quantitative perspective that offers new insights into the interplay between LLM creativity and hallucination. The code and data for our experiments are available at https://github.com/ZicongHe2002/HCL-Spark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02846v1">Mask-DPO: Generalizable Fine-grained Factuality Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted by ICLR 2025. Code is available at https://github.com/open-compass/ANAH
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit hallucinations (i.e., unfaithful or nonsensical information) when serving as AI assistants in various domains. Since hallucinations always come with truthful content in the LLM responses, previous factuality alignment methods that conduct response-level preference learning inevitably introduced noises during training. Therefore, this paper proposes a fine-grained factuality alignment method based on Direct Preference Optimization (DPO), called Mask-DPO. Incorporating sentence-level factuality as mask signals, Mask-DPO only learns from factually correct sentences in the preferred samples and prevents the penalty on factual contents in the not preferred samples, which resolves the ambiguity in the preference learning. Extensive experimental results demonstrate that Mask-DPO can significantly improve the factuality of LLMs responses to questions from both in-domain and out-of-domain datasets, although these questions and their corresponding topics are unseen during training. Only trained on the ANAH train set, the score of Llama3.1-8B-Instruct on the ANAH test set is improved from 49.19% to 77.53%, even surpassing the score of Llama3.1-70B-Instruct (53.44%), while its FactScore on the out-of-domain Biography dataset is also improved from 30.29% to 39.39%. We further study the generalization property of Mask-DPO using different training sample scaling strategies and find that scaling the number of topics in the dataset is more effective than the number of questions. We provide a hypothesis of what factual alignment is doing with LLMs, on the implication of this phenomenon, and conduct proof-of-concept experiments to verify it. We hope the method and the findings pave the way for future research on scaling factuality alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21239v2">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 This paper needs approval from Amazon for open resource release
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring white-box access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02800v1">RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.00914
    </div>
    <details class="paper-abstract">
      Anomaly detection in complex industrial environments poses unique challenges, particularly in contexts characterized by data sparsity and evolving operational conditions. Predictive maintenance (PdM) in such settings demands methodologies that are adaptive, transferable, and capable of integrating domain-specific knowledge. In this paper, we present RAAD-LLM, a novel framework for adaptive anomaly detection, leveraging large language models (LLMs) integrated with Retrieval-Augmented Generation (RAG). This approach addresses the aforementioned PdM challenges. By effectively utilizing domain-specific knowledge, RAAD-LLM enhances the detection of anomalies in time series data without requiring fine-tuning on specific datasets. The framework's adaptability mechanism enables it to adjust its understanding of normal operating conditions dynamically, thus increasing detection accuracy. We validate this methodology through a real-world application for a plastics manufacturing plant and the Skoltech Anomaly Benchmark (SKAB). Results show significant improvements over our previous model with an accuracy increase from 70.7 to 89.1 on the real-world dataset. By allowing for the enriching of input series data with semantics, RAAD-LLM incorporates multimodal capabilities that facilitate more collaborative decision-making between the model and plant operators. Overall, our findings support RAAD-LLM's ability to revolutionize anomaly detection methodologies in PdM, potentially leading to a paradigm shift in how anomaly detection is implemented across various industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01711v2">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02776v1">Implicit Bias in LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Due to the implement of guardrails by developers, Large language models (LLMs) have demonstrated exceptional performance in explicit bias tests. However, bias in LLMs may occur not only explicitly, but also implicitly, much like humans who consciously strive for impartiality yet still harbor implicit bias. The unconscious and automatic nature of implicit bias makes it particularly challenging to study. This paper provides a comprehensive review of the existing literature on implicit bias in LLMs. We begin by introducing key concepts, theories and methods related to implicit bias in psychology, extending them from humans to LLMs. Drawing on the Implicit Association Test (IAT) and other psychological frameworks, we categorize detection methods into three primary approaches: word association, task-oriented text generation and decision-making. We divide our taxonomy of evaluation metrics for implicit bias into two categories: single-value-based metrics and comparison-value-based metrics. We classify datasets into two types: sentences with masked tokens and complete sentences, incorporating datasets from various domains to reflect the broad application of LLMs. Although research on mitigating implicit bias in LLMs is still limited, we summarize existing efforts and offer insights on future challenges. We aim for this work to serve as a clear guide for researchers and inspire innovative ideas to advance exploration in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00320v2">Shifting Power: Leveraging LLMs to Simulate Human Aversion in ABMs of Bilateral Financial Exchanges, A bond market study</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Bilateral markets, such as those for government bonds, involve decentralized and opaque transactions between market makers (MMs) and clients, posing significant challenges for traditional modeling approaches. To address these complexities, we introduce TRIBE an agent-based model augmented with a large language model (LLM) to simulate human-like decision-making in trading environments. TRIBE leverages publicly available data and stylized facts to capture realistic trading dynamics, integrating human biases like risk aversion and ambiguity sensitivity into the decision-making processes of agents. Our research yields three key contributions: first, we demonstrate that integrating LLMs into agent-based models to enhance client agency is feasible and enriches the simulation of agent behaviors in complex markets; second, we find that even slight trade aversion encoded within the LLM leads to a complete cessation of trading activity, highlighting the sensitivity of market dynamics to agents' risk profiles; third, we show that incorporating human-like variability shifts power dynamics towards clients and can disproportionately affect the entire system, often resulting in systemic agent collapse across simulations. These findings underscore the emergent properties that arise when introducing stochastic, human-like decision processes, revealing new system behaviors that enhance the realism and complexity of artificial societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05821v2">Towards Zero-Shot, Controllable Dialog Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 This paper has been accepted for publication at the AAAI 2022 Workshop on Planning in the Era of LLMs
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have emerged as an alternative to training task-specific dialog agents, due to their broad reasoning capabilities and performance in zero-shot learning scenarios. However, many LLM-based dialog systems fall short in planning towards an overarching dialog goal and therefore cannot steer the conversation appropriately. Furthermore, these models struggle with hallucination, making them unsuitable for information access in sensitive domains, such as legal or medical domains, where correctness of information given to users is critical. The recently introduced task Conversational Tree Search (CTS) proposes the use of dialog graphs to avoid hallucination in sensitive domains, however, state-of-the-art agents are Reinforcement Learning (RL) based and require long training times, despite excelling at dialog strategy. This paper introduces a novel zero-shot method for controllable CTS agents, where LLMs guide the dialog planning through domain graphs by searching and pruning relevant graph nodes based on user interaction preferences. We show that these agents significantly outperform state-of-the-art CTS agents ($p<0.0001$; Barnard Exact test) in simulation. This generalizes to all available CTS domains. Finally, we perform user evaluation to test the agent's performance in the wild, showing that our policy significantly ($p<0.05$; Barnard Exact) improves task-success compared to the state-of-the-art RL-based CTS agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02718v1">Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Understanding the semantics of columns in relational tables is an important pre-processing step for indexing data lakes in order to provide rich data search. An approach to establishing such understanding is column type annotation (CTA) where the goal is to annotate table columns with terms from a given vocabulary. This paper experimentally compares different knowledge generation and self-refinement strategies for LLM-based column type annotation. The strategies include using LLMs to generate term definitions, error-based refinement of term definitions, self-correction, and fine-tuning using examples and term definitions. We evaluate these strategies along two dimensions: effectiveness measured as F1 performance and efficiency measured in terms of token usage and cost. Our experiments show that the best performing strategy depends on the model/dataset combination. We find that using training data to generate label definitions outperforms using the same data as demonstrations for in-context learning for two out of three datasets using OpenAI models. The experiments further show that using the LLMs to refine label definitions brings an average increase of 3.9% F1 in 10 out of 12 setups compared to the performance of the non-refined definitions. Combining fine-tuned models with self-refined term definitions results in the overall highest performance, outperforming zero-shot prompting fine-tuned models by at least 3% in F1 score. The costs analysis shows that while reaching similar F1 score, self-refinement via prompting is more cost efficient for use cases requiring smaller amounts of tables to be annotated while fine-tuning is more efficient for large amounts of tables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02698v1">FlowPlan: Zero-Shot Task Planning with LLM Flow Engineering for Robotic Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Robotic instruction following tasks require seamless integration of visual perception, task planning, target localization, and motion execution. However, existing task planning methods for instruction following are either data-driven or underperform in zero-shot scenarios due to difficulties in grounding lengthy instructions into actionable plans under operational constraints. To address this, we propose FlowPlan, a structured multi-stage LLM workflow that elevates zero-shot pipeline and bridges the performance gap between zero-shot and data-driven in-context learning methods. By decomposing the planning process into modular stages--task information retrieval, language-level reasoning, symbolic-level planning, and logical evaluation--FlowPlan generates logically coherent action sequences while adhering to operational constraints and further extracts contextual guidance for precise instance-level target localization. Benchmarked on the ALFRED and validated in real-world applications, our method achieves competitive performance relative to data-driven in-context learning methods and demonstrates adaptability across diverse environments. This work advances zero-shot task planning in robotic systems without reliance on labeled data. Project website: https://instruction-following-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v3">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02682v1">MPO: Boosting LLM Agents with Meta Plan Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled LLM-based agents to successfully tackle interactive planning tasks. However, despite their successes, existing approaches often suffer from planning hallucinations and require retraining for each new agent. To address these challenges, we propose the Meta Plan Optimization (MPO) framework, which enhances agent planning capabilities by directly incorporating explicit guidance. Unlike previous methods that rely on complex knowledge, which either require significant human effort or lack quality assurance, MPO leverages high-level general guidance through meta plans to assist agent planning and enables continuous optimization of the meta plans based on feedback from the agent's task execution. Our experiments conducted on two representative tasks demonstrate that MPO significantly outperforms existing baselines. Moreover, our analysis indicates that MPO provides a plug-and-play solution that enhances both task completion efficiency and generalization capabilities in previous unseen scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02628v1">Towards Event Extraction with Massive Types: LLM-based Collaborative Annotation and Partitioning Extraction</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Developing a general-purpose extraction system that can extract events with massive types is a long-standing target in Event Extraction (EE). In doing so, the challenge comes from two aspects: 1) The absence of an efficient and effective annotation method. 2) The absence of a powerful extraction method can handle massive types. For the first challenge, we propose a collaborative annotation method based on Large Language Models (LLMs). Through collaboration among multiple LLMs, it first refines annotations of trigger words from distant supervision and then carries out argument annotation. Next, a voting phase consolidates the annotation preferences across different LLMs. Finally, we create the EEMT dataset, the largest EE dataset to date, featuring over 200,000 samples, 3,465 event types, and 6,297 role types. For the second challenge, we propose an LLM-based Partitioning EE method called LLM-PEE. To overcome the limited context length of LLMs, LLM-PEE first recalls candidate event types and then splits them into multiple partitions for LLMs to extract events. The results in the supervised setting show that LLM-PEE outperforms the state-of-the-art methods by 5.4 in event detection and 6.1 in argument extraction. In the zero-shot setting, LLM-PEE achieves up to 12.9 improvement compared to mainstream LLMs, demonstrating its strong generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04070v6">PAD: Personalized Alignment of LLMs at Decoding-Time</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Aligning with personalized preferences, which vary significantly across cultural, educational, and political differences, poses a significant challenge due to the computational costs and data demands of traditional alignment methods. In response, this paper presents Personalized Alignment at Decoding-time (PAD), a novel framework designed to align LLM outputs with diverse personalized preferences during the inference phase, eliminating the need for additional training. By introducing a unique personalized reward modeling strategy, this framework decouples the text generation process from personalized preferences, facilitating the generation of generalizable token-level personalized rewards. The PAD algorithm leverages these rewards to guide the decoding process, dynamically tailoring the base model's predictions to personalized preferences. Extensive experimental results demonstrate that PAD not only outperforms existing training-based alignment methods in terms of aligning with diverse preferences but also shows significant generalizability to preferences unseen during training and scalability across different base models. This work advances the capability of LLMs to meet user needs in real-time applications, presenting a substantial step forward in personalized LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02597v1">Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent Multimodal Large Language Models (MLLMs) have demonstrated significant progress in perceiving and reasoning over multimodal inquiries, ushering in a new research era for foundation models. However, vision-language misalignment in MLLMs has emerged as a critical challenge, where the textual responses generated by these models are not factually aligned with the given text-image inputs. Existing efforts to address vision-language misalignment have focused on developing specialized vision-language connectors or leveraging visual instruction tuning from diverse domains. In this paper, we tackle this issue from a fundamental yet unexplored perspective by revisiting the core architecture of MLLMs. Most MLLMs are typically built on decoder-only LLMs consisting of a causal attention mechanism, which limits the ability of earlier modalities (e.g., images) to incorporate information from later modalities (e.g., text). To address this problem, we propose AKI, a novel MLLM that unlocks causal attention into modality-mutual attention (MMA) to enable image tokens to attend to text tokens. This simple yet effective design allows AKI to achieve superior performance in 12 multimodal understanding benchmarks (+7.2% on average) without introducing additional parameters and increasing training time. Our MMA design is intended to be generic, allowing for application across various modalities, and scalable to accommodate diverse multimodal scenarios. The code is publicly available at https://github.com/sony/aki, and we will release our AKI-4B model to encourage further advancements in MLLMs across various directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00946v2">A Review of LLM-Assisted Ideation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      We present a comprehensive, in-depth review of ideation assisted by large language models (LLMs), highlighting emerging trends and identifying unaddressed research gaps. In total, we examined 61 studies investigating the application of LLMs in both group and individual ideation processes. From these studies, we derived the Hourglass Ideation Framework for LLM-assisted ideation, comprising three phases and seven key ideation stages, which served as the basis for our systematic survey. Our analysis reveals that LLMs are most frequently used for idea generation and refinement, but their use in scope specification, foundational material structuring and multi-idea evaluation and selection remains limited. We provide our findings in extensive tabular and online formats. These catalogues detail research on LLM-assisted, purely LLM-based, and human-only activities across the seven ideation stages for each of the 61 studies. These also detail creative domains, publication outlets, interaction designs, user study designs, and assessment methods. Our analysis of system interaction design reveals a predominant focus on supporting individual ideation activities and text-based interaction, with a growing trend of incorporating multimedia elements. However, in group ideation, tools and interaction modalities targeting both synchronous and asynchronous collaboration are much scarcer. We synthesize the primary findings of our review and outline promising directions for future research in LLM-assisted ideation. We hope this review will help researchers quickly gain an overview of this rapidly expanding area, efficiently locate relevant work, and identify underexplored areas for further investigation. In addition, we believe the framework we present here will form the basis for the development of future problem and solution space taxonomies, and methodologies for LLM-assisted ideation development and use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03157v2">Let the Code LLM Edit Itself When You Edit the Code</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 ICLR 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      In this work, we investigate a typical scenario in code generation where a developer edits existing code in real time and requests a code assistant, e.g., a large language model, to re-predict the next token or next line on the fly. Naively, the LLM needs to re-encode the entire KV cache to provide an accurate prediction. However, this process is computationally expensive, especially when the sequence length is long. Simply encoding the edited subsequence and integrating it to the original KV cache meets the temporal confusion problem, leading to significantly worse performance. We address this efficiency and accuracy trade-off by introducing \underline{\textbf{Positional \textbf{I}ntegrity \textbf{E}ncoding} (PIE). Building upon the rotary positional encoding, PIE first removes the rotary matrices in the Key cache that introduce temporal confusion and then reapplies the correct rotary matrices. This process ensures that positional relationships between tokens are correct and requires only a single round of matrix multiplication. We validate the effectiveness of PIE through extensive experiments on the RepoBench-C-8k dataset, utilizing DeepSeek-Coder models with 1.3B, 6.7B, and 33B parameters. Our evaluation includes three real-world coding tasks: code insertion, code deletion, and multi-place code editing. Results demonstrate that PIE reduces computational overhead by over 85% compared to the standard full recomputation approach across all model sizes and tasks while well approximating the model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01622v2">DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation. Browse the data, contribute, and more: https://slab-nlp.github.io/DOVE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02574v1">LLM-Safety Evaluations Lack Robustness</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      In this paper, we argue that current safety alignment research efforts for large language models are hindered by many intertwined sources of noise, such as small datasets, methodological inconsistencies, and unreliable evaluation setups. This can, at times, make it impossible to evaluate and compare attacks and defenses fairly, thereby slowing progress. We systematically analyze the LLM safety evaluation pipeline, covering dataset curation, optimization strategies for automated red-teaming, response generation, and response evaluation using LLM judges. At each stage, we identify key issues and highlight their practical impact. We also propose a set of guidelines for reducing noise and bias in evaluations of future attack and defense papers. Lastly, we offer an opposing perspective, highlighting practical reasons for existing limitations. We believe that addressing the outlined problems in future research will improve the field's ability to generate easily comparable results and make measurable progress.
    </details>
</div>
