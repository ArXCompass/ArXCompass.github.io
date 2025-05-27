# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16502v1">Recursive Offloading for LLM Serving in Multi-tier Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Heterogeneous device-edge-cloud computing infrastructures have become widely adopted in telecommunication operators and Wide Area Networks (WANs), offering multi-tier computational support for emerging intelligent services. With the rapid proliferation of Large Language Model (LLM) services, efficiently coordinating inference tasks and reducing communication overhead within these multi-tier network architectures becomes a critical deployment challenge. Existing LLM serving paradigms exhibit significant limitations: on-device deployment supports only lightweight LLMs due to hardware constraints, while cloud-centric deployment suffers from resource congestion and considerable prompt communication overhead caused by frequent service requests during peak periods. Although the model-cascading-based inference strategy adapts better to multi-tier networks, its reliance on fine-grained, manually adjusted thresholds makes it less responsive to dynamic network conditions and varying task complexities. To address these challenges, we propose RecServe, a recursive offloading framework tailored for LLM serving in multi-tier networks. RecServe integrates a task-specific hierarchical confidence evaluation mechanism that guides offloading decisions based on inferred task complexity in progressively scaled LLMs across device, edge, and cloud tiers. To further enable intelligent task routing across tiers, RecServe employs a sliding-window-based dynamic offloading strategy with quantile interpolation, enabling real-time tracking of historical confidence distributions and adaptive offloading threshold adjustments. Experiments on eight datasets demonstrate that RecServe outperforms CasServe in both service quality and communication efficiency, and reduces the communication burden by over 50\% compared to centralized cloud-based serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15701v2">When LLMs Learn to be Students: The SOEI Framework for Modeling and Evaluating Virtual Student Agents in Educational Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled intelligent tutoring systems, yet the development of LLM-based Virtual Student Agents (LVSAs) remains underexplored. Such agents are essential for teacher-facing applications, where simulating diverse learner traits can support adaptive instruction and pedagogical skill development. However, current methods lack principled personality modeling, scalable evaluation of behavioral consistency, and empirical validation in interactive teaching settings. We propose the SOEI framework, a structured pipeline comprising Scene, Object, Evaluation, and Interaction, for constructing and evaluating personality-aligned LVSAs in classroom scenarios. Leveraging Chinese language instruction as a cognitively and emotionally rich testbed, we generate five LVSAs based on Big Five traits through LoRA fine-tuning and expert-informed prompt design. Their behavioral realism and personality coherence are assessed using a hybrid human & GPT-4 evaluation and a multi-dimensional annotation protocol. Through controlled experiments with real pre-service teachers, we demonstrate that LVSAs can elicit adaptive teaching strategies and maintain trait-consistent behavior across multi-turn dialogues. Our results provide: (1) an educationally and psychologically grounded generation pipeline for LLM-based student agents; (2) a hybrid, scalable evaluation framework for behavioral realism; and (3) empirical insights into the pedagogical utility of LVSAs in shaping instructional adaptation. By embedding LVSAs into both generative modeling and human-in-the-loop teaching, SOEI bridges AI for Education (AI4Edu) and Education for AI (Edu4AI), positioning classroom interaction as a rigorous testbed for controllability, personality alignment, and human-likeness in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v3">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Updated version, with examples in the appendix
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation as being superior to human translation from experienced professionals. In the long run, this bias could result in an irreversible decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LiTransProQA, a novel, reference-free, LLM-based question-answering framework designed for literary translation evaluation. LiTransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LiTransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LiTransProQA reaches human-level evaluation performance comparable to trained student evaluators. It shows broad applicability to open-source models like LLaMa3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free tool for evaluating literary translations that require local processing due to copyright or ethical considerations. The code and datasets are available under: https://github.com/zhangr2021/TransProQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16467v1">Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06846v2">Prot2Chat: Protein LLM with Early-Fusion of Text, Sequence and Structure</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Motivation: Proteins are of great significance in living organisms. However, understanding their functions encounters numerous challenges, such as insufficient integration of multimodal information, a large number of training parameters, limited flexibility of classification-based methods, and the lack of systematic evaluation metrics for protein Q&A systems. To tackle these issues, we propose the Prot2Chat framework. Results: We modified ProteinMPNN to encode protein sequence and structural information in a unified way. We used a large language model (LLM) to encode questions into vectors and developed a protein-text adapter to compress protein information into virtual tokens based on these vectors, achieving the early fusion of text and protein information. Finally, the same LLM reads the virtual tokens and the questions to generate answers. To optimize training efficiency, we froze the encoder and employed Low-Rank Adaptation (LoRA) techniques for the LLM. Experiments on two datasets show that both automated metrics and expert evaluations demonstrate the superior performance of our model, and zero-shot prediction results highlight its generalization ability. The models and codes are available at https://github.com/ wangzc1233/Prot2Chat. Contact: zqcao@suda.edu.cn or wangzc025@163.com Key words: Protein Q&A, Early-Fusion, LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11617v2">ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 9 pages, multiple figures
    </div>
    <details class="paper-abstract">
      Assembly code analysis and comprehension play critical roles in applications like reverse engineering, yet they face substantial challenges due to low information density and a lack of explicit syntactic structures. While traditional masked language modeling (MLM) approaches do not explicitly focus on natural language interaction, emerging decoder-focused large language models (LLMs) demonstrate partial success in binary analysis yet remain underexplored for holistic comprehension. We present Assembly Augmented Tuning, an end-to-end structural-semantic instruction tuning framework that synergizes encoder architecture with decoder-based LLMs through a projector module, where the assembly encoder extracts hardware-level structural features, the projector bridges representations with the semantic space, and the instruction-tuned LLM preserves natural language capabilities. Experimental results demonstrate three key advantages: (1) State-of-the-art performance in assembly comprehension with +39.7% Recall@1 and +17.8% MRR improvements over GPT-4-Turbo, (2) Consistent enhancements across base models (24.6-107.4% Recall@1 and 15.2-106.3% MRR on Qwen2.5-Coder, Deepseek-Coder and CodeLlama variants), and (3) Superior instruction-following capabilities (41.5%-118% improvements) with controlled code generation degradation (-8.9% to -35% across architectures).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16456v1">MAGIC: Motion-Aware Generative Inference via Confidence-Guided LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in static 3D generation have intensified the demand for physically consistent dynamic 3D content. However, existing video generation models, including diffusion-based methods, often prioritize visual realism while neglecting physical plausibility, resulting in implausible object dynamics. Prior approaches for physics-aware dynamic generation typically rely on large-scale annotated datasets or extensive model fine-tuning, which imposes significant computational and data collection burdens and limits scalability across scenarios. To address these challenges, we present MAGIC, a training-free framework for single-image physical property inference and dynamic generation, integrating pretrained image-to-video diffusion models with iterative LLM-based reasoning. Our framework generates motion-rich videos from a static image and closes the visual-to-physical gap through a confidence-driven LLM feedback loop that adaptively steers the diffusion model toward physics-relevant motion. To translate visual dynamics into controllable physical behavior, we further introduce a differentiable MPM simulator operating directly on 3D Gaussians reconstructed from the single image, enabling physically grounded, simulation-ready outputs without any supervision or model tuning. Experiments show that MAGIC outperforms existing physics-aware generative methods in inference accuracy and achieves greater temporal coherence than state-of-the-art video diffusion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16455v1">Psychology-driven LLM Agents for Explainable Panic Prediction on Social Media during Sudden Disaster Events</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      During sudden disaster events, accurately predicting public panic sentiment on social media is crucial for proactive governance and crisis management. Current efforts on this problem face three main challenges: lack of finely annotated data hinders emotion prediction studies, unmodeled risk perception causes prediction inaccuracies, and insufficient interpretability of panic formation mechanisms. We address these issues by proposing a Psychology-driven generative Agent framework (PsychoAgent) for explainable panic prediction based on emotion arousal theory. Specifically, we first construct a fine-grained open panic emotion dataset (namely COPE) via human-large language models (LLMs) collaboration to mitigate semantic bias. Then, we develop a framework integrating cross-domain heterogeneous data grounded in psychological mechanisms to model risk perception and cognitive differences in emotion generation. To enhance interpretability, we design an LLM-based role-playing agent that simulates individual psychological chains through dedicatedly designed prompts. Experimental results on our annotated dataset show that PsychoAgent improves panic emotion prediction performance by 12.6% to 21.7% compared to baseline models. Furthermore, the explainability and generalization of our approach is validated. Crucially, this represents a paradigm shift from opaque "data-driven fitting" to transparent "role-based simulation with mechanistic interpretation" for panic emotion prediction during emergencies. Our implementation is publicly available at: https://anonymous.4open.science/r/PsychoAgent-19DD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16410v1">Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at https://github.com/dongguanting/Tool-Star.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16408v1">From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06086v2">Is a Peeled Apple Still Red? Evaluating LLMs' Ability for Conceptual Combination with Property Type</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 NAACL 2025 Oral
    </div>
    <details class="paper-abstract">
      Conceptual combination is a cognitive process that merges basic concepts, enabling the creation of complex expressions. During this process, the properties of combination (e.g., the whiteness of a peeled apple) can be inherited from basic concepts, newly emerge, or be canceled. However, previous studies have evaluated a limited set of properties and have not examined the generative process. To address this gap, we introduce the Conceptual Combination with Property Type dataset (CCPT), which consists of 12.3K annotated triplets of noun phrases, properties, and property types. Using CCPT, we establish three types of tasks to evaluate LLMs for conceptual combination thoroughly. Our key findings are threefold: (1) Our automatic metric grading property emergence and cancellation closely corresponds with human judgments. (2) LLMs, including OpenAI's o1, struggle to generate noun phrases which possess given emergent properties. (3) Our proposed method, inspired by cognitive psychology model that explains how relationships between concepts are formed, improves performances in all generative tasks. The dataset and experimental code are available at https://github.com/seokwon99/CCPT.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16363v1">AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We introduce AdamS, a simple yet effective alternative to Adam for large language model (LLM) pretraining and post-training. By leveraging a novel denominator, i.e., the root of weighted sum of squares of the momentum and the current gradient, AdamS eliminates the need for second-moment estimates. Hence, AdamS is efficient, matching the memory and compute footprint of SGD with momentum while delivering superior optimization performance. Moreover, AdamS is easy to adopt: it can directly inherit hyperparameters of AdamW, and is entirely model-agnostic, integrating seamlessly into existing pipelines without modifications to optimizer APIs or architectures. The motivation behind AdamS stems from the observed $(L_0, L_1)$ smoothness properties in transformer objectives, where local smoothness is governed by gradient magnitudes that can be further approximated by momentum magnitudes. We establish rigorous theoretical convergence guarantees and provide practical guidelines for hyperparameter selection. Empirically, AdamS demonstrates strong performance in various tasks, including pre-training runs on GPT-2 and Llama2 (up to 13B parameters) and reinforcement learning in post-training regimes. With its efficiency, simplicity, and theoretical grounding, AdamS stands as a compelling alternative to existing optimizers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16340v1">Improving Chemical Understanding of LLMs via SMILES Parsing</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly recognized as powerful tools for scientific discovery, particularly in molecular science. A fundamental requirement for these models is the ability to accurately understand molecular structures, commonly encoded in the SMILES representation. However, current LLMs struggle to interpret SMILES, even failing to carry out basic tasks such as counting molecular rings. To address this limitation, we introduce CLEANMOL, a novel framework that formulates SMILES parsing into a suite of clean and deterministic tasks explicitly designed to promote graph-level molecular comprehension. These tasks span from subgraph matching to global graph matching, providing structured supervision aligned with molecular structural properties. We construct a molecular pretraining dataset with adaptive difficulty scoring and pre-train open-source LLMs on these tasks. Our results show that CLEANMOL not only enhances structural comprehension but also achieves the best or competes with the baseline on the Mol-Instructions benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16339v1">Rethinking Code Review Workflows with LLM Assistance: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Code reviews are a critical yet time-consuming aspect of modern software development, increasingly challenged by growing system complexity and the demand for faster delivery. This paper presents a study conducted at WirelessCar Sweden AB, combining an exploratory field study of current code review practices with a field experiment involving two variations of an LLM-assisted code review tool. The field study identifies key challenges in traditional code reviews, including frequent context switching, insufficient contextual information, and highlights both opportunities (e.g., automatic summarization of complex pull requests) and concerns (e.g., false positives and trust issues) in using LLMs. In the field experiment, we developed two prototype variations: one offering LLM-generated reviews upfront and the other enabling on-demand interaction. Both utilize a semantic search pipeline based on retrieval-augmented generation to assemble relevant contextual information for the review, thereby tackling the uncovered challenges. Developers evaluated both variations in real-world settings: AI-led reviews are overall more preferred, while still being conditional on the reviewers' familiarity with the code base, as well as on the severity of the pull request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16312v1">EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at https://github.com/Lolo1222/EquivPruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v3">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by ACL2025(Findings)
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16303v1">INFERENCEDYNAMICS: Efficient Routing Across LLMs through Structured Capability and Knowledge Profiling</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) routing is a pivotal technique for navigating a diverse landscape of LLMs, aiming to select the best-performing LLMs tailored to the domains of user queries, while managing computational resources. However, current routing approaches often face limitations in scalability when dealing with a large pool of specialized LLMs, or in their adaptability to extending model scope and evolving capability domains. To overcome those challenges, we propose InferenceDynamics, a flexible and scalable multi-dimensional routing framework by modeling the capability and knowledge of models. We operate it on our comprehensive dataset RouteMix, and demonstrate its effectiveness and generalizability in group-level routing using modern benchmarks including MMLU-Pro, GPQA, BigGenBench, and LiveBench, showcasing its ability to identify and leverage top-performing models for given tasks, leading to superior outcomes with efficient resource utilization. The broader adoption of Inference Dynamics can empower users to harness the full specialized potential of the LLM ecosystem, and our code will be made publicly available to encourage further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16293v1">Augmenting LLM Reasoning with Dynamic Notes Writing for Complex QA</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Iterative RAG for multi-hop question answering faces challenges with lengthy contexts and the buildup of irrelevant information. This hinders a model's capacity to process and reason over retrieved content and limits performance. While recent methods focus on compressing retrieved information, they are either restricted to single-round RAG, require finetuning or lack scalability in iterative RAG. To address these challenges, we propose Notes Writing, a method that generates concise and relevant notes from retrieved documents at each step, thereby reducing noise and retaining only essential information. This indirectly increases the effective context length of Large Language Models (LLMs), enabling them to reason and plan more effectively while processing larger volumes of input text. Notes Writing is framework agnostic and can be integrated with different iterative RAG methods. We demonstrate its effectiveness with three iterative RAG methods, across two models and four evaluation datasets. Notes writing yields an average improvement of 15.6 percentage points overall, with minimal increase in output tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16277v1">Spontaneous Speech Variables for Evaluating LLMs Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 The 14th Workshop on Cognitive Modeling and Computational Linguistics (CMCL). May 3, 2025. Collocated with NAACL 2025
    </div>
    <details class="paper-abstract">
      The achievements of Large Language Models in Natural Language Processing, especially for high-resource languages, call for a better understanding of their characteristics from a cognitive perspective. Researchers have attempted to evaluate artificial models by testing their ability to predict behavioral (e.g., eye-tracking fixations) and physiological (e.g., brain responses) variables during language processing (e.g., reading/listening). In this paper, we propose using spontaneous speech corpora to derive production variables (speech reductions, prosodic prominences) and applying them in a similar fashion. More precisely, we extract. We then test models trained with a standard procedure on different pretraining datasets (written, spoken, and mixed genres) for their ability to predict these two variables. Our results show that, after some fine-tuning, the models can predict these production variables well above baselines. We also observe that spoken genre training data provides more accurate predictions than written genres. These results contribute to the broader effort of using high-quality speech corpora as benchmarks for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v2">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06139v2">LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to NAACL 2025. Project Page: https://ssuminan.github.io/LCIRC/
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) excel in generating coherent and contextually rich outputs, their capacity to efficiently handle long-form contexts is limited by fixed-length position embeddings. Additionally, the computational cost of processing long sequences increases quadratically, making it challenging to extend context length. To address these challenges, we propose Long-form Context Injection with Recurrent Compression (LCIRC), a method that enables the efficient processing long-form sequences beyond the model's length limit through recurrent compression without retraining the entire model. We further introduce query dependent context modeling, which selectively compresses query-relevant information, ensuring that the model retains the most pertinent content. Our empirical results demonstrate that Query Dependent LCIRC (QD-LCIRC) significantly improves LLM's ability to manage extended contexts, making it well-suited for tasks that require both comprehensive context understanding and query relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16254v1">Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs: What Still Applies and What We Must Leave Behind</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      In this paper, we conduct a critical review of existing theories and frameworks on human-human collaborative writing to assess their relevance to the current human-AI paradigm in professional contexts, and draw seven insights along with design implications for human-AI collaborative writing tools. We found that, as LLMs nudge the writing process more towards an empirical "trial and error" process analogous to prototyping, the non-linear cognitive process of writing will stay the same, but more rigor will be required for revision methodologies. This shift would shed further light on the importance of coherence support, but the large language model (LLM)'s unprecedented semantic capabilities can bring novel approaches to this ongoing challenge. We argue that teamwork-related factors such as group awareness, consensus building and authorship - which have been central in human-human collaborative writing studies - should not apply to the human-AI paradigm due to excessive anthropomorphism. With the LLM's text generation capabilities becoming essentially indistinguishable from human-written ones, we are entering an era where, for the first time in the history of computing, we are engaging in collaborative writing with AI at workplaces on a daily basis. We aim to bring theoretical grounding and practical design guidance to the interaction designs of human-AI collaborative writing, with the goal of enhancing future human-AI writing software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16222v1">Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to both positive and negative biases, resulting in inflated or unfairly low scores. Moreover, we observe that LLM judges remain vulnerable to these biases even when prompted to generate test cases before scoring, highlighting the need for more robust code evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15793v2">HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to hallucinations. Evaluations show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16221v1">LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models has unlocked remarkable capabilities across a diverse array of natural language processing tasks. However, the considerable differences among available LLMs-in terms of cost, performance, and computational demands-pose significant challenges for users aiming to identify the most suitable model for specific tasks. In this work, we present LightRouter, a novel framework designed to systematically select and integrate a small subset of LLMs from a larger pool, with the objective of jointly optimizing both task performance and cost efficiency. LightRouter leverages an adaptive selection mechanism to identify models that require only a minimal number of boot tokens, thereby reducing costs, and further employs an effective integration strategy to combine their outputs. Extensive experiments across multiple benchmarks demonstrate that LightRouter matches or outperforms widely-used ensemble baselines, achieving up to a 25% improvement in accuracy. Compared with leading high-performing models, LightRouter achieves comparable performance while reducing inference costs by up to 27%. Importantly, our framework operates without any prior knowledge of individual models and relies exclusively on inexpensive, lightweight models. This work introduces a practical approach for efficient LLM selection and provides valuable insights into optimal strategies for model combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13830v2">Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by Interspeech 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16216v1">Memorization or Reasoning? Exploring the Idiom Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06272v2">Evaluating LLM-based Approaches to Legal Citation Prediction: Domain-specific Pre-training, Fine-tuning, or RAG? A Benchmark and an Australian Law Case Study</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 For code, data, and models see https://auslawbench.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong potential across legal tasks, yet the problem of legal citation prediction remains under-explored. At its core, this task demands fine-grained contextual understanding and precise identification of relevant legislation or precedent. We introduce the AusLaw Citation Benchmark, a real-world dataset comprising 55k Australian legal instances and 18,677 unique citations which to the best of our knowledge is the first of its scale and scope. We then conduct a systematic benchmarking across a range of solutions: (i) standard prompting of both general and law-specialised LLMs, (ii) retrieval-only pipelines with both generic and domain-specific embeddings, (iii) supervised fine-tuning, and (iv) several hybrid strategies that combine LLMs with retrieval augmentation through query expansion, voting ensembles, or re-ranking. Results show that neither general nor law-specific LLMs suffice as stand-alone solutions, with performance near zero. Instruction tuning (of even a generic open-source LLM) on task-specific dataset is among the best performing solutions. We highlight that database granularity along with the type of embeddings play a critical role in retrieval-based approaches, with hybrid methods which utilise a trained re-ranker delivering the best results. Despite this, a performance gap of nearly 50% remains, underscoring the value of this challenging benchmark as a rigorous test-bed for future research in legal-domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13862v2">PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09285v2">DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      LLM inference must meet strict latency SLOs (e.g., 100 ms P99 time-between-tokens) while maximizing goodput. Yet, real-world variability in prompt and response lengths skews compute-intensive prefill and memory-bound decode phases, making both colocated (even with chunked prefill) and disaggregated deployments unable to simultaneously deliver low tail latency and high throughput. We introduce DynaServe, a high-performance LLM serving system built atop vLLM that unifies and extends both paradigms for maximizing goodput under SLO constraints, when handling unbalanced and dynamic workloads. It relies on a micro-request abstraction, which arbitrarily splits each request at any token boundary into at most two cooperating segments. A two-level scheduling framework then balances micro-request load across unified GPU instances. The global scheduler rapidly selects per-request split points by considering both the request's prefill/decode time ratio and the current load across GPU instances. The local schedulers on each GPU instance independently form SLO-aware batches, adjusting their composition in response to workload fluctuations, potential latency spikes and per-GPU under/over utilization. On real-world traces, DynaServe boosts the overall serving capacity from 1.15$\times$ to 3.07$\times$, improves goodput by up to 1.91$\times$ and 1.61$\times$, and improves the performance by up to 60\% in a hybrid workload under SLO compared to state-of-the-art colocated and disaggregated baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16170v1">When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on https://github.com/ayyyq/llm-retraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16164v1">Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 model configurations, varying prompt specificity, sampling temperature, and model type, and compared outputs to responses from 106 human participants. While some configurations, especially Claude 3.7 Sonnet, matched human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse and structurally rigid, and LLM ensembles failed to increase diversity. Network analyses further revealed fundamental differences in retrieval structure between humans and models. These results highlight key limitations in using LLMs to simulate human cognition and behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16142v1">Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16129v1">LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 5 pages, 2 figures, 2 tables. Conforms to the ACL Rolling Review (ARR) short paper track. Code and data available at: https://github.com/CuiNiki/LLMs-Are-Not-Scorers
    </div>
    <details class="paper-abstract">
      Recent studies have applied large language models (LLMs) to machine translation quality estimation (MTQE) by prompting models to assign numeric scores. Nonetheless, these direct scoring methods tend to show low segment-level correlation with human judgments. In this paper, we propose a generation-based evaluation paradigm that leverages decoder-only LLMs to produce high-quality references, followed by semantic similarity scoring using sentence embeddings. We conduct the most extensive evaluation to date in MTQE, covering 8 LLMs and 8 language pairs. Empirical results show that our method outperforms both intra-LLM direct scoring baselines and external non-LLM reference-free metrics from MTME. These findings demonstrate the strength of generation-based evaluation and support a shift toward hybrid approaches that combine fluent generation with accurate semantic assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16128v1">Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Despite LLMs' explicit alignment against demographic stereotypes, they have been shown to exhibit biases under various social contexts. In this work, we find that LLMs exhibit concerning biases in how they associate solution veracity with demographics. Through experiments across five human value-aligned LLMs on mathematics, coding, commonsense, and writing problems, we reveal two forms of such veracity biases: Attribution Bias, where models disproportionately attribute correct solutions to certain demographic groups, and Evaluation Bias, where models' assessment of identical solutions varies based on perceived demographic authorship. Our results show pervasive biases: LLMs consistently attribute fewer correct solutions and more incorrect ones to African-American groups in math and coding, while Asian authorships are least preferred in writing evaluation. In additional studies, we show LLMs automatically assign racially stereotypical colors to demographic groups in visualization code, suggesting these biases are deeply embedded in models' reasoning processes. Our findings indicate that demographic bias extends beyond surface-level stereotypes and social context provocations, raising concerns about LLMs' deployment in educational and evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16120v1">LLM-Powered AI Agent Systems and Their Applications in Industry</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 This is the author's accepted version of the paper accepted to appear at IEEE AIIoT 2025. The final version will be available via IEEE Xplore. \c{opyright}2025 IEEE. Personal use of this material is permitted
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16118v1">Semiotic Reconstruction of Destination Expectation Constructs An LLM-Driven Computational Paradigm for Social Media Tourism Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 33 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Social media's rise establishes user-generated content (UGC) as pivotal for travel decisions, yet analytical methods lack scalability. This study introduces a dual-method LLM framework: unsupervised expectation extraction from UGC paired with survey-informed supervised fine-tuning. Findings reveal leisure/social expectations drive engagement more than foundational natural/emotional factors. By establishing LLMs as precision tools for expectation quantification, we advance tourism analytics methodology and propose targeted strategies for experience personalization and social travel promotion. The framework's adaptability extends to consumer behavior research, demonstrating computational social science's transformative potential in marketing optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16113v1">Tools in the Loop: Quantifying Uncertainty of LLM Question Answering Systems That Use Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 10 pages 3 figures 3 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) often require external tools, such as machine learning classifiers or knowledge retrieval systems, to provide accurate answers in domains where their pre-trained knowledge is insufficient. This integration of LLMs with external tools expands their utility but also introduces a critical challenge: determining the trustworthiness of responses generated by the combined system. In high-stakes applications, such as medical decision-making, it is essential to assess the uncertainty of both the LLM's generated text and the tool's output to ensure the reliability of the final response. However, existing uncertainty quantification methods do not account for the tool-calling scenario, where both the LLM and external tool contribute to the overall system's uncertainty. In this work, we present a novel framework for modeling tool-calling LLMs that quantifies uncertainty by jointly considering the predictive uncertainty of the LLM and the external tool. We extend previous methods for uncertainty quantification over token sequences to this setting and propose efficient approximations that make uncertainty computation practical for real-world applications. We evaluate our framework on two new synthetic QA datasets, derived from well-known machine learning datasets, which require tool-calling for accurate answers. Additionally, we apply our method to retrieval-augmented generation (RAG) systems and conduct a proof-of-concept experiment demonstrating the effectiveness of our uncertainty metrics in scenarios where external information retrieval is needed. Our results show that the framework is effective in enhancing trust in LLM-based systems, especially in cases where the LLM's internal knowledge is insufficient and external tools are required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01917v2">Steer LLM Latents for Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12459v2">LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to PAKDD 2025
    </div>
    <details class="paper-abstract">
      Topic modeling is widely used for uncovering thematic structures within text corpora, yet traditional models often struggle with specificity and coherence in domain-focused applications. Guided approaches, such as SeededLDA and CorEx, incorporate user-provided seed words to improve relevance but remain labor-intensive and static. Large language models (LLMs) offer potential for dynamic topic refinement and discovery, yet their application often incurs high API costs. To address these challenges, we propose the LLM-assisted Iterative Topic Augmentation framework (LITA), an LLM-assisted approach that integrates user-provided seeds with embedding-based clustering and iterative refinement. LITA identifies a small number of ambiguous documents and employs an LLM to reassign them to existing or new topics, minimizing API costs while enhancing topic quality. Experiments on two datasets across topic quality and clustering performance metrics demonstrate that LITA outperforms five baseline models, including LDA, SeededLDA, CorEx, BERTopic, and PromptTopic. Our work offers an efficient and adaptable framework for advancing topic modeling and text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16090v1">Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 6 pages, 4 figures. Research conducted as part of a Microsoft-sponsored Capstone Project at Santa Clara University
    </div>
    <details class="paper-abstract">
      As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16086v1">Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17332v1">SweEval: Do LLMs Really Swear? A Safety Benchmark for Testing Limits for Enterprise Use</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Published in the Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2025), Industry Track, pages 558-582
    </div>
    <details class="paper-abstract">
      Enterprise customers are increasingly adopting Large Language Models (LLMs) for critical communication tasks, such as drafting emails, crafting sales pitches, and composing casual messages. Deploying such models across different regions requires them to understand diverse cultural and linguistic contexts and generate safe and respectful responses. For enterprise applications, it is crucial to mitigate reputational risks, maintain trust, and ensure compliance by effectively identifying and handling unsafe or offensive language. To address this, we introduce SweEval, a benchmark simulating real-world scenarios with variations in tone (positive or negative) and context (formal or informal). The prompts explicitly instruct the model to include specific swear words while completing the task. This benchmark evaluates whether LLMs comply with or resist such inappropriate instructions and assesses their alignment with ethical frameworks, cultural nuances, and language comprehension capabilities. In order to advance research in building ethically aligned AI systems for enterprise use and beyond, we release the dataset and code: https://github.com/amitbcp/multilingual_profanity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11228v2">Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 A RAG pipeline that accounts for both diversity and answer quality and that can be used with any LLM backbone to solve complex multi-hop question-answering tasks
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) for domain-specific question-answering (QA) tasks by leveraging external knowledge sources. However, traditional RAG systems primarily focus on relevance-based retrieval and often struggle with redundancy, especially when reasoning requires connecting information from multiple sources. This paper introduces Vendi-RAG, a framework based on an iterative process that jointly optimizes retrieval diversity and answer quality. This joint optimization leads to significantly higher accuracy for multi-hop QA tasks. Vendi-RAG leverages the Vendi Score (VS), a flexible similarity-based diversity metric, to promote semantic diversity in document retrieval. It then uses an LLM judge that evaluates candidate answers, generated after a reasoning step, and outputs a score that the retriever uses to balance relevance and diversity among the retrieved documents during each iteration. Experiments on three challenging datasets -- HotpotQA, MuSiQue, and 2WikiMultiHopQA -- demonstrate Vendi-RAG's effectiveness in multi-hop reasoning tasks. The framework achieves significant accuracy improvements over traditional single-step and multi-step RAG approaches, with accuracy increases reaching up to +4.2% on HotpotQA, +4.1% on 2WikiMultiHopQA, and +1.3% on MuSiQue compared to Adaptive-RAG, the current best baseline. The benefits of Vendi-RAG are even more pronounced as the number of retrieved documents increases. Finally, we evaluated Vendi-RAG across different LLM backbones, including GPT-3.5, GPT-4, and GPT-4o-mini, and observed consistent improvements, demonstrating that the framework's advantages are model-agnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17327v1">GPT Editors, Not Authors: The Stylistic Footprint of LLMs in Academic Preprints</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) in late 2022 has impacted academic writing, threatening credibility, and causing institutional uncertainty. We seek to determine the degree to which LLMs are used to generate critical text as opposed to being used for editing, such as checking for grammar errors or inappropriate phrasing. In our study, we analyze arXiv papers for stylistic segmentation, which we measure by varying a PELT threshold against a Bayesian classifier trained on GPT-regenerated text. We find that LLM-attributed language is not predictive of stylistic segmentation, suggesting that when authors use LLMs, they do so uniformly, reducing the risk of hallucinations being introduced into academic preprints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05522v3">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ACL'25 (Industry) Oral
    </div>
    <details class="paper-abstract">
      Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential solutions by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. To enhance planning for new user interests using LLMs, this paper introduces a novel approach that combines hierarchical planning with LLM inference-time scaling. This method aims to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17265v1">CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Rare diseases, including Inborn Errors of Metabolism (IEM), pose significant diagnostic challenges. Case reports serve as key but computationally underutilized resources to inform diagnosis. Clinical dense information extraction refers to organizing medical information into structured predefined categories. Large Language Models (LLMs) may enable scalable information extraction from case reports but are rarely evaluated for this task. We introduce CaseReportBench, an expert-annotated dataset for dense information extraction of case reports, focusing on IEMs. Using this dataset, we assess various models and prompting strategies, introducing novel approaches such as category-specific prompting and subheading-filtered data integration. Zero-shot chain-of-thought prompting offers little advantage over standard zero-shot prompting. Category-specific prompting improves alignment with the benchmark. The open-source model Qwen2.5-7B outperforms GPT-4o for this task. Our clinician evaluations show that LLMs can extract clinically relevant details from case reports, supporting rare disease diagnosis and management. We also highlight areas for improvement, such as LLMs' limitations in recognizing negative findings important for differential diagnosis. This work advances LLM-driven clinical natural language processing and paves the way for scalable medical AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17249v1">Where You Go is Who You Are: Behavioral Theory-Guided LLMs for Inverse Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Big trajectory data hold great promise for human mobility analysis, but their utility is often constrained by the absence of critical traveler attributes, particularly sociodemographic information. While prior studies have explored predicting such attributes from mobility patterns, they often overlooked underlying cognitive mechanisms and exhibited low predictive accuracy. This study introduces SILIC, short for Sociodemographic Inference with LLM-guided Inverse Reinforcement Learning (IRL) and Cognitive Chain Reasoning (CCR), a theoretically grounded framework that leverages LLMs to infer sociodemographic attributes from observed mobility patterns by capturing latent behavioral intentions and reasoning through psychological constructs. Particularly, our approach explicitly follows the Theory of Planned Behavior (TPB), a foundational behavioral framework in transportation research, to model individuals' latent cognitive processes underlying travel decision-making. The LLMs further provide heuristic guidance to improve IRL reward function initialization and update by addressing its ill-posedness and optimization challenges arising from the vast and unstructured reward space. Evaluated in the 2017 Puget Sound Regional Council Household Travel Survey, our method substantially outperforms state-of-the-art baselines and shows great promise for enriching big trajectory data to support more behaviorally grounded applications in transportation planning and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13794v3">LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17217v1">Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17196v1">Shape it Up! Restoring LLM Safety during Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Finetuning large language models (LLMs) enables user-specific customization but introduces critical safety risks: even a few harmful examples can compromise safety alignment. A common mitigation strategy is to update the model more strongly on examples deemed safe, while downweighting or excluding those flagged as unsafe. However, because safety context can shift within a single example, updating the model equally on both harmful and harmless parts of a response is suboptimal-a coarse treatment we term static safety shaping. In contrast, we propose dynamic safety shaping (DSS), a framework that uses fine-grained safety signals to reinforce learning from safe segments of a response while suppressing unsafe content. To enable such fine-grained control during finetuning, we introduce a key insight: guardrail models, traditionally used for filtering, can be repurposed to evaluate partial responses, tracking how safety risk evolves throughout the response, segment by segment. This leads to the Safety Trajectory Assessment of Response (STAR), a token-level signal that enables shaping to operate dynamically over the training sequence. Building on this, we present STAR-DSS, guided by STAR scores, that robustly mitigates finetuning risks and delivers substantial safety improvements across diverse threats, datasets, and model families-all without compromising capability on intended tasks. We encourage future safety research to build on dynamic shaping principles for stronger mitigation against evolving finetuning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17169v1">Next Token Perception Score: Analytical Assessment of your LLM Perception Skills</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Autoregressive pretraining has become the de facto paradigm for learning general-purpose representations in large language models (LLMs). However, linear probe performance across downstream perception tasks shows substantial variability, suggesting that features optimized for next-token prediction do not consistently transfer well to downstream perception tasks. We demonstrate that representations learned via autoregression capture features that may lie outside the subspaces most informative for perception. To quantify the (mis)alignment between autoregressive pretraining and downstream perception, we introduce the Next Token Perception Score (NTPS)-a score derived under a linear setting that measures the overlap between autoregressive and perception feature subspaces. This metric can be easily computed in closed form from pretrained representations and labeled data, and is proven to both upper- and lower-bound the excess loss. Empirically, we show that NTPS correlates strongly with linear probe accuracy across 12 diverse NLP datasets and eight pretrained models ranging from 270M to 8B parameters, confirming its utility as a measure of alignment. Furthermore, we show that NTPS increases following low-rank adaptation (LoRA) fine-tuning, especially in large models, suggesting that LoRA aligning representations to perception tasks enhances subspace overlap and thus improves downstream performance. More importantly, we find that NTPS reliably predicts the additional accuracy gains attained by LoRA finetuning thereby providing a lightweight prescreening tool for LoRA adaptation. Our results offer both theoretical insights and practical tools for analytically assessing LLM perception skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17162v1">DailyQA: A Benchmark to Evaluate Web Retrieval Augmented LLMs Based on Capturing Real-World Changes</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We propose DailyQA, an automatically updated dynamic dataset that updates questions weekly and contains answers to questions on any given date. DailyQA utilizes daily updates from Wikipedia revision logs to implement a fully automated pipeline of data filtering, query generation synthesis, quality checking, answer extraction, and query classification. The benchmark requires large language models (LLMs) to process and answer questions involving fast-changing factual data and covering multiple domains. We evaluate several open-source and closed-source LLMs using different RAG pipelines with web search augmentation. We compare the ability of different models to process time-sensitive web information and find that rerank of web retrieval results is critical. Our results indicate that LLMs still face significant challenges in handling frequently updated information, suggesting that DailyQA benchmarking provides valuable insights into the direction of progress for LLMs and RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17156v1">PersonaBOT: Bringing Customer Personas to Life with LLMs and RAG</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar's test. In the final step, the chatbot's knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17148v1">LLM-Powered Agents for Navigating Venice's Historical Cadastre</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Cadastral data reveal key information about the historical organization of cities but are often non-standardized due to diverse formats and human annotations, complicating large-scale analysis. We explore as a case study Venice's urban history during the critical period from 1740 to 1808, capturing the transition following the fall of the ancient Republic and the Ancien R\'egime. This era's complex cadastral data, marked by its volume and lack of uniform structure, presents unique challenges that our approach adeptly navigates, enabling us to generate spatial queries that bridge past and present urban landscapes. We present a text-to-programs framework that leverages Large Language Models (LLMs) to translate natural language queries into executable code for processing historical cadastral records. Our methodology implements two complementary techniques: a text-to-SQL approach for handling structured queries about specific cadastral information, and a text-to-Python approach for complex analytical operations requiring custom data manipulation. We propose a taxonomy that classifies historical research questions based on their complexity and analytical requirements, mapping them to the most appropriate technical approach. This framework is supported by an investigation into the execution consistency of the system, alongside a qualitative analysis of the answers it produces. By ensuring interpretability and minimizing hallucination through verifiable program outputs, we demonstrate the system's effectiveness in reconstructing past population information, property features, and spatiotemporal comparisons in Venice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17147v1">MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 19 pages,6 figures,ACL2025
    </div>
    <details class="paper-abstract">
      The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17145v1">LLM Access Shield: Domain-Specific LLM Framework for Privacy Policy Compliance</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in fields such as finance, education, and governance due to their ability to generate human-like text and adapt to specialized tasks. However, their widespread adoption raises critical concerns about data privacy and security, including the risk of sensitive data exposure. In this paper, we propose a security framework to enforce policy compliance and mitigate risks in LLM interactions. Our approach introduces three key innovations: (i) LLM-based policy enforcement: a customizable mechanism that enhances domain-specific detection of sensitive data. (ii) Dynamic policy customization: real-time policy adaptation and enforcement during user-LLM interactions to ensure compliance with evolving security requirements. (iii) Sensitive data anonymization: a format-preserving encryption technique that protects sensitive information while maintaining contextual integrity. Experimental results demonstrate that our framework effectively mitigates security risks while preserving the functional accuracy of LLM-driven tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17140v1">Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 4 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As the knowledge of large language models (LLMs) becomes outdated over time, there is a growing need for efficient methods to update them, especially when injecting proprietary information. Our study reveals that comprehension-intensive fine-tuning tasks (e.g., question answering and blanks) achieve substantially higher knowledge retention rates (48%) compared to mapping-oriented tasks like translation (17%) or text-to-JSON conversion (20%), despite exposure to identical factual content. We demonstrate that this pattern persists across model architectures and follows scaling laws, with larger models showing improved retention across all task types. However, all models exhibit significant performance drops when applying injected knowledge in broader contexts, suggesting limited semantic integration. These findings show the importance of task selection in updating LLM knowledge, showing that effective knowledge injection relies not just on data exposure but on the depth of cognitive engagement during fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17138v1">RAP: Runtime-Adaptive Pruning for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v1">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Submitted to the IEEE GlobeCom 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v1">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17131v1">Relative Bias: A Comparative Framework for Quantifying Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The growing deployment of large language models (LLMs) has amplified concerns regarding their inherent biases, raising critical questions about their fairness, safety, and societal impact. However, quantifying LLM bias remains a fundamental challenge, complicated by the ambiguity of what "bias" entails. This challenge grows as new models emerge rapidly and gain widespread use, while introducing potential biases that have not been systematically assessed. In this paper, we propose the Relative Bias framework, a method designed to assess how an LLM's behavior deviates from other LLMs within a specified target domain. We introduce two complementary methodologies: (1) Embedding Transformation analysis, which captures relative bias patterns through sentence representations over the embedding space, and (2) LLM-as-a-Judge, which employs a language model to evaluate outputs comparatively. Applying our framework to several case studies on bias and alignment scenarios following by statistical tests for validation, we find strong alignment between the two scoring methods, offering a systematic, scalable, and statistically grounded approach for comparative bias analysis in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15444v1">Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Existing studies have optimized retrieval-augmented generation (RAG) across various sub-tasks, such as query understanding and retrieval refinement, but integrating these optimizations into a unified framework remains challenging. To tackle this problem, this work proposes RoleRAG, a unified RAG framework that achieves efficient multi-task processing through role-specific token optimization. RoleRAG comprises six modules, each handling a specific sub-task within the RAG process. Additionally, we introduce a query graph to represent the decomposition of the query, which can be dynamically resolved according to the decomposing state. All modules are driven by the same underlying LLM, distinguished by task-specific role tokens that are individually optimized. This design allows RoleRAG to dynamically activate different modules within a single LLM instance, thereby streamlining deployment and reducing resource consumption. Experimental results on five open-domain question-answering datasets demonstrate the effectiveness, generalizability, and flexibility of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15433v1">Set-LLM: A Permutation-Invariant LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) demonstrate impressive capabilities across numerous applications, their robustness remains a critical concern. This paper is motivated by a specific vulnerability: the order sensitivity of LLMs. This vulnerability manifests itself as the order bias observed when LLMs decide between possible options (for example, a preference for the first option) and the tendency of LLMs to provide different answers when options are reordered. The use cases for this scenario extend beyond the classical case of multiple-choice question answering to the use of LLMs as automated evaluators in AI pipelines, comparing output generated by different models. We introduce Set-LLM, a novel architectural adaptation for pretrained LLMs that enables the processing of mixed set-text inputs with permutation invariance guarantees. The adaptations involve a new attention mask and new positional encodings specifically designed for sets. We provide a theoretical proof of invariance and demonstrate through experiments that Set-LLM can be trained effectively, achieving comparable or improved performance and maintaining the runtime of the original model, while eliminating order sensitivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15426v1">NeoN: A Tool for Automated Detection, Linguistic and LLM-Driven Analysis of Neologisms in Polish</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 15 pages, this is an extended version of a paper accepted for the 25th International Conference on Computational Science (ICCS), 7-9 July 2025
    </div>
    <details class="paper-abstract">
      NeoN, a tool for detecting and analyzing Polish neologisms. Unlike traditional dictionary-based methods requiring extensive manual review, NeoN combines reference corpora, Polish-specific linguistic filters, an LLM-driven precision-boosting filter, and daily RSS monitoring in a multi-layered pipeline. The system uses context-aware lemmatization, frequency analysis, and orthographic normalization to extract candidate neologisms while consolidating inflectional variants. Researchers can verify candidates through an intuitive interface with visualizations and filtering controls. An integrated LLM module automatically generates definitions and categorizes neologisms by domain and sentiment. Evaluations show NeoN maintains high accuracy while significantly reducing manual effort, providing an accessible solution for tracking lexical innovation in Polish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15422v1">Trends and Challenges in Authorship Analysis: A Review of ML, DL, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 25 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Authorship analysis plays an important role in diverse domains, including forensic linguistics, academia, cybersecurity, and digital content authentication. This paper presents a systematic literature review on two key sub-tasks of authorship analysis; Author Attribution and Author Verification. The review explores SOTA methodologies, ranging from traditional ML approaches to DL models and LLMs, highlighting their evolution, strengths, and limitations, based on studies conducted from 2015 to 2024. Key contributions include a comprehensive analysis of methods, techniques, their corresponding feature extraction techniques, datasets used, and emerging challenges in authorship analysis. The study highlights critical research gaps, particularly in low-resource language processing, multilingual adaptation, cross-domain generalization, and AI-generated text detection. This review aims to help researchers by giving an overview of the latest trends and challenges in authorship analysis. It also points out possible areas for future study. The goal is to support the development of better, more reliable, and accurate authorship analysis system in diverse textual domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15410v1">ClickSight: Interpreting Student Clickstreams to Reveal Insights on Learning Strategies via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted in Latebreaking results track in AIED 2025(26th International Conference on Artificial Intelligence in Education JULY 22-26, 2025 PALERMO, ITALY)
    </div>
    <details class="paper-abstract">
      Clickstream data from digital learning environments offer valuable insights into students' learning behaviors, but are challenging to interpret due to their high dimensionality and granularity. Prior approaches have relied mainly on handcrafted features, expert labeling, clustering, or supervised models, therefore often lacking generalizability and scalability. In this work, we introduce ClickSight, an in-context Large Language Model (LLM)-based pipeline that interprets student clickstreams to reveal their learning strategies. ClickSight takes raw clickstreams and a list of learning strategies as input and generates textual interpretations of students' behaviors during interaction. We evaluate four different prompting strategies and investigate the impact of self-refinement on interpretation quality. Our evaluation spans two open-ended learning environments and uses a rubric-based domain-expert evaluation. Results show that while LLMs can reasonably interpret learning strategies from clickstreams, interpretation quality varies by prompting strategy, and self-refinement offers limited improvement. ClickSight demonstrates the potential of LLMs to generate theory-driven insights from educational interaction data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15392v1">An Empirical Study of the Anchoring Effect in LLMs: Existence, Mechanism, and Potential Mitigations</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) like ChatGPT has advanced natural language processing, yet concerns about cognitive biases are growing. In this paper, we investigate the anchoring effect, a cognitive bias where the mind relies heavily on the first information as anchors to make affected judgments. We explore whether LLMs are affected by anchoring, the underlying mechanisms, and potential mitigation strategies. To facilitate studies at scale on the anchoring effect, we introduce a new dataset, SynAnchors. Combining refined evaluation metrics, we benchmark current widely used LLMs. Our findings show that LLMs' anchoring bias exists commonly with shallow-layer acting and is not eliminated by conventional strategies, while reasoning can offer some mitigation. This recontextualization via cognitive psychology urges that LLM evaluations focus not on standard benchmarks or over-optimized robustness tests, but on cognitive-bias-aware trustworthy evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15365v1">AI vs. Human Judgment of Content Moderation: LLM-as-a-Judge and Ethics-Based Response Refusals</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in high-stakes settings, their ability to refuse ethically sensitive prompts-such as those involving hate speech or illegal activities-has become central to content moderation and responsible AI practices. While refusal responses can be viewed as evidence of ethical alignment and safety-conscious behavior, recent research suggests that users may perceive them negatively. At the same time, automated assessments of model outputs are playing a growing role in both evaluation and training. In particular, LLM-as-a-Judge frameworks-in which one model is used to evaluate the output of another-are now widely adopted to guide benchmarking and fine-tuning. This paper examines whether such model-based evaluators assess refusal responses differently than human users. Drawing on data from Chatbot Arena and judgments from two AI judges (GPT-4o and Llama 3 70B), we compare how different types of refusals are rated. We distinguish ethical refusals, which explicitly cite safety or normative concerns (e.g., "I can't help with that because it may be harmful"), and technical refusals, which reflect system limitations (e.g., "I can't answer because I lack real-time data"). We find that LLM-as-a-Judge systems evaluate ethical refusals significantly more favorably than human users, a divergence not observed for technical refusals. We refer to this divergence as a moderation bias-a systematic tendency for model-based evaluators to reward refusal behaviors more than human users do. This raises broader questions about transparency, value alignment, and the normative assumptions embedded in automated evaluation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v2">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 41 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01941v2">Can LLMs Maintain Fundamental Abilities under KV Cache Compression?</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      This paper investigates an underexplored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. Although existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive benchmark KVFundaBench to systematically evaluate the effects of KV cache compression across diverse fundamental LLM capabilities, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals serval key findings: (1) \textit{Task-Dependent Degradation}; (2) \textit{Model-Type Robustness} (3) \textit{Prompt Length Vulnerability}; (4) \textit{Chunk-Level Superiority}; (5) \textit{Prompt-Gain Sensitivity}; (6) \textit{Long-Context Generation Sensitivity}. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v3">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15347v1">FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15337v1">Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15323v1">Improving LLM First-Token Predictions in Multiple-Choice Question Answering via Prefilling Attack</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 13 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly evaluated on multiple-choice question answering (MCQA) tasks using *first-token probability* (FTP), which selects the answer option whose initial token has the highest likelihood. While efficient, FTP can be fragile: models may assign high probability to unrelated tokens (*misalignment*) or use a valid token merely as part of a generic preamble rather than as a clear answer choice (*misinterpretation*), undermining the reliability of symbolic evaluation. We propose a simple solution: the *prefilling attack*, a structured natural-language prefix (e.g., "*The correct option is:*") prepended to the model output. Originally explored in AI safety, we repurpose prefilling to steer the model to respond with a clean, valid option, without modifying its parameters. Empirically, the FTP with prefilling strategy substantially improves accuracy, calibration, and output consistency across a broad set of LLMs and MCQA benchmarks. It outperforms standard FTP and often matches the performance of open-ended generation approaches that require full decoding and external classifiers, while being significantly more efficient. Our findings suggest that prefilling is a simple, robust, and low-cost method to enhance the reliability of FTP-based evaluation in multiple-choice settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15311v1">Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v3">Automate Strategy Finding with LLM in Quant Investment</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      We present a novel three-stage framework leveraging Large Language Models (LLMs) within a risk-aware multi-agent system for automate strategy finding in quantitative finance. Our approach addresses the brittleness of traditional deep learning models in financial applications by: employing prompt-engineered LLMs to generate executable alpha factor candidates across diverse financial data, implementing multimodal agent-based evaluation that filters factors based on market status, predictive quality while maintaining category balance, and deploying dynamic weight optimization that adapts to market conditions. Experimental results demonstrate the robust performance of the strategy in Chinese & US market regimes compared to established benchmarks. Our work extends LLMs capabilities to quantitative trading, providing a scalable architecture for financial signal extraction and portfolio construction. The overall framework significantly outperforms all benchmarks with 53.17% cumulative return on SSE50 (Jan 2023 to Jan 2024), demonstrating superior risk-adjusted performance and downside protection on the market.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14350v3">An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given within a certain output length. It is unclear whether and how the reasoning ability of different LLMs remain effective under strict constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test 30 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between token budgets and actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning ability that differ from the unconstrained situation, e.g. the optimal choices of either model size or prompt style change under different budgets. These findings offer timely evaluation to this area and practical guidance for users to deploy LLMs under real-world latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15257v1">When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 26 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-source LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively decoupled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training such as supervised fine-tuning or reinforcement learning, our training-free ablation achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15240v1">Generalised Probabilistic Modelling and Improved Uncertainty Estimation in Comparative LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 To appear in UAI 2025
    </div>
    <details class="paper-abstract">
      This paper explores generalised probabilistic modelling and uncertainty estimation in comparative LLM-as-a-judge frameworks. We show that existing Product-of-Experts methods are specific cases of a broader framework, enabling diverse modelling options. Furthermore, we propose improved uncertainty estimates for individual comparisons, enabling more efficient selection and achieving strong performance with fewer evaluations. We also introduce a method for estimating overall ranking uncertainty. Finally, we demonstrate that combining absolute and comparative scoring improves performance. Experiments show that the specific expert model has a limited impact on final rankings but our proposed uncertainty estimates, especially the probability of reordering, significantly improve the efficiency of systems reducing the number of needed comparisons by ~50%. Furthermore, ranking-level uncertainty metrics can be used to identify low-performing predictions, where the nature of the probabilistic model has a notable impact on the quality of the overall uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15229v1">Multilingual Prompting for Improving LLM Generation Diversity</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to lack cultural representation and overall diversity in their generations, from expressing opinions to answering factual questions. To mitigate this problem, we propose multilingual prompting: a prompting method which generates several variations of a base prompt with added cultural and linguistic cues from several cultures, generates responses, and then combines the results. Building on evidence that LLMs have language-specific knowledge, multilingual prompting seeks to increase diversity by activating a broader range of cultural knowledge embedded in model training data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA 70B, and LLaMA 8B), we show that multilingual prompting consistently outperforms existing diversity-enhancing techniques such as high-temperature sampling, step-by-step recall, and personas prompting. Further analyses show that the benefits of multilingual prompting vary with language resource level and model size, and that aligning the prompting language with the cultural cues reduces hallucination about culturally-specific information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05179v2">Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 15 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 78% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02145v3">From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 New version of the paper
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14552v2">KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09049v3">Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Discovering customer intentions in dialogue conversations is crucial for automated service agents. However, existing intent clustering methods often fail to align with human perceptions due to a heavy reliance on embedding distance metrics and a tendency to overlook underlying semantic structures. This paper proposes an LLM-in-the-loop (LLM-ITL) intent clustering framework, integrating the semantic understanding capabilities of LLMs into conventional clustering algorithms. Specifically, this paper (1) investigates the effectiveness of fine-tuned LLMs in semantic coherence evaluation and intent cluster naming, achieving over 95% accuracy aligned with human judgments; (2) designs an LLM-ITL framework that facilitates the iterative discovery of coherent intent clusters and the optimal number of clusters; and (3) proposes context-aware techniques tailored for customer service dialogue. As existing English benchmarks offer limited semantic diversity and intent groups, we introduce a comprehensive Chinese dialogue intent dataset, comprising over 100k real customer service calls and 1,507 human-annotated intent clusters. The proposed approaches significantly outperform LLM-guided baselines, achieving notable enhancements in clustering quality and lower computational cost. Combined with several best practices, our findings highlight the potential of LLM-in-the-loop techniques for scalable and human-aligned intent clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15182v1">ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02009v2">Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 10 pages, 5 figures. Accepted at the International Joint Conferences on Artificial Intelligence IJCAI 2025 (main track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for harmful content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. We share TTP, TTP-Eval, HAVOC and a sample of C4 inferenced on HarmFormer. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15154v1">Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advancements in reasoning have significantly enhanced the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse tasks. However, excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs, reducing efficiency. Our work reveals that prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks. To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel framework that dynamically switches between short answers and long-form reasoning based on the model perplexity. CAR first generates a short answer and evaluates its perplexity, triggering reasoning only when the model exhibits low confidence (i.e., high perplexity). Experiments across diverse multimodal VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04295v3">Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant capability across various tasks, with their real-world effectiveness often driven by prompt design. While recent research has focused on optimizing prompt content, the role of prompt formatting, a critical but often overlooked dimension, has received limited systematic investigation. In this paper, we introduce Content-Format Integrated Prompt Optimization (CFPO), an innovative methodology that jointly optimizes both prompt content and formatting through an iterative refinement process. CFPO leverages natural language mutations to explore content variations and employs a dynamic format exploration strategy that systematically evaluates diverse format options. Our extensive evaluations across multiple tasks and open-source LLMs demonstrate that CFPO demonstrates measurable performance improvements compared to content-only optimization methods. This highlights the importance of integrated content-format optimization and offers a practical, model-agnostic approach to enhancing LLM performance. Code is available at https://github.com/HenryLau7/CFPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15146v1">lmgame-Bench: How Good are LLMs at Playing Games?</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Playing video games requires perception, memory, and planning, exactly the faculties modern large language model (LLM) agents are expected to master. We study the major challenges in using popular video games to evaluate modern LLMs and find that directly dropping LLMs into games cannot make an effective evaluation, for three reasons -- brittle vision perception, prompt sensitivity, and potential data contamination. We introduce lmgame-Bench to turn games into reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and narrative games delivered through a unified Gym-style API and paired with lightweight perception and memory scaffolds, and is designed to stabilize prompt variance and remove contamination. Across 13 leading models, we show lmgame-Bench is challenging while still separating models well. Correlation analysis shows that every game probes a unique blend of capabilities often tested in isolation elsewhere. More interestingly, performing reinforcement learning on a single game from lmgame-Bench transfers both to unseen games and to external planning tasks. Our evaluation code is available at https://github.com/lmgame-org/GamingAgent/lmgame-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20302v2">A Multilingual, Culture-First Approach to Addressing Misgendering in LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Misgendering is the act of referring to someone by a gender that does not match their chosen identity. It marginalizes and undermines a person's sense of self, causing significant harm. English-based approaches have clear-cut approaches to avoiding misgendering, such as the use of the pronoun ``they''. However, other languages pose unique challenges due to both grammatical and cultural constructs. In this work we develop methodologies to assess and mitigate misgendering across 42 languages and dialects using a participatory-design approach to design effective and appropriate guardrails across all languages. We test these guardrails in a standard LLM-based application (meeting transcript summarization), where both the data generation and the annotation steps followed a human-in-the-loop approach. We find that the proposed guardrails are very effective in reducing misgendering rates across all languages in the summaries generated, and without incurring loss of quality. Our human-in-the-loop approach demonstrates a method to feasibly scale inclusive and responsible AI-based solutions across multiple languages and cultures. We release the guardrails and synthetic dataset encompassing 42 languages, along with human and LLM-judge evaluations, to encourage further research on this subject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15134v1">The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Entropy minimization (EM) trains the model to concentrate even more probability mass on its most confident outputs. We show that this simple objective alone, without any labeled data, can substantially improve large language models' (LLMs) performance on challenging math, physics, and coding tasks. We explore three approaches: (1) EM-FT minimizes token-level entropy similarly to instruction finetuning, but on unlabeled outputs drawn from the model; (2) EM-RL: reinforcement learning with negative entropy as the only reward to maximize; (3) EM-INF: inference-time logit adjustment to reduce entropy without any training data or parameter updates. On Qwen-7B, EM-RL, without any labeled data, achieves comparable or better performance than strong RL baselines such as GRPO and RLOO that are trained on 60K labeled examples. Furthermore, EM-INF enables Qwen-32B to match or exceed the performance of proprietary models like GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro on the challenging SciCode benchmark, while being 3x more efficient than self-consistency and sequential refinement. Our findings reveal that many pretrained LLMs possess previously underappreciated reasoning capabilities that can be effectively elicited through entropy minimization alone, without any labeled data or even any parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05849v2">AGENTFUZZER: Generic Black-Box Fuzzing for Indirect Prompt Injection against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11023v2">Beyond A Single AI Cluster: A Survey of Decentralized LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has revolutionized AI development, yet the resource demands beyond a single cluster or even datacenter, limiting accessibility to well-resourced organizations. Decentralized training has emerged as a promising paradigm to leverage dispersed resources across clusters, datacenters and regions, offering the potential to democratize LLM development for broader communities. As the first comprehensive exploration of this emerging field, we present decentralized LLM training as a resource-driven paradigm and categorize existing efforts into community-driven and organizational approaches. We further clarify this through: (1) a comparison with related paradigms, (2) a characterization of decentralized resources, and (3) a taxonomy of recent advancements. We also provide up-to-date case studies and outline future directions to advance research in decentralized LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v7">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15117v1">An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15107v1">StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at https://github.com/zxh20001117/StepSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v2">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15101v1">Cost-aware LLM-based Online Dataset Annotation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled automated dataset labeling with minimal human supervision. While majority voting across multiple LLMs can improve label reliability by mitigating individual model biases, it incurs high computational costs due to repeated querying. In this work, we propose a novel online framework, Cost-aware Majority Voting (CaMVo), for efficient and accurate LLM-based dataset annotation. CaMVo adaptively selects a subset of LLMs for each data instance based on contextual embeddings, balancing confidence and cost without requiring pre-training or ground-truth labels. Leveraging a LinUCB-based selection mechanism and a Bayesian estimator over confidence scores, CaMVo estimates a lower bound on labeling accuracy for each LLM and aggregates responses through weighted majority voting. Our empirical evaluation on the MMLU and IMDB Movie Review datasets demonstrates that CaMVo achieves comparable or superior accuracy to full majority voting while significantly reducing labeling costs. This establishes CaMVo as a practical and robust solution for cost-efficient annotation in dynamic labeling environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03797v3">NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The resurgence of autonomous agents built using large language models (LLMs) to solve complex real-world tasks has brought increased focus on LLMs' fundamental ability of tool or function calling. At the core of these agents, an LLM must plan, execute, and respond using external tools, APIs, and custom functions. Research on tool calling has gathered momentum, but evaluation benchmarks and datasets representing the complexity of the tasks have lagged behind. In this work, we focus on one such complexity, nested sequencing, with the goal of extending existing benchmarks and evaluation. Specifically, we present NESTFUL, a benchmark to evaluate LLMs on nested sequences of API calls, i.e., sequences where the output of one API call is passed as input to a subsequent call. NESTFUL contains 1800+ nested sequences where all the function calls are executable. Experimental results on a variety of models show that the best-performing model (GPT-4o) achieves a full sequence match accuracy of 28% and a win-rate of 60%, necessitating a large scope for improvement in the nested sequencing aspect of function calling. Our analysis of these results provides possible future research directions for the community, in addition to a benchmark to track progress. We have released the NESTFUL dataset under the Apache 2.0 license at https://github.com/IBM/NESTFUL.
    </details>
</div>
